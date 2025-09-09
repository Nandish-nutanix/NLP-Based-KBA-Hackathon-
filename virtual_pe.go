package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
	arithmos_client "github.com/nutanix-core/go-cache/stats/arithmos/interface"
	arithmos_pr "github.com/nutanix-core/go-cache/stats/arithmos"
	insights_interface "github.com/nutanix-core/go-cache/insights/insights_interface"
	"github.com/golang/protobuf/proto"
)

const (
	PYTHON_VIRTUALENV_PATH = "/home/nutanix/.venvs/bin/bin/python3.9"
	CHUNK_SIZE             = 1000

	LOG_ERROR   = 0
	LOG_WARNING = 1
	LOG_INFO    = 2
	LOG_DEBUG   = 3
	LOG_TRACE   = 4
	LOG_VM      = 5
)

var (
	peIP                   string
	vmNamePrefix           string
	numVMs                 int
	numVCPUs               string
	memory                 string
	taskCount              int
	taskPeriodSecs         int
	peIPsFile              string
	diskSize               string
	diskCount              int
	vnicCount              int
	vmBatchCount           int
	powerOnFilter          bool
	dataPointVariations    int
	alertsPerMinute        int
	containerCount         int
	isPhysicalCluster      bool
	skipDeploy             bool
	periodicNumVMsToCreate int
	sleepInterval          int
	logLevel               int
	statsLogInterval       int
	showTiming             bool
)

var MapEntityTypeToEnum = map[string]int{
	"kDisk":             1,
	"kVDisk":            2,
	"kVM":               3,
	"kContainer":        4,
	"kStoragePool":      5,
	"kNode":             6,
	"kCluster":          7,
	"kVirtualDisk":      8,
	"kRemoteSite":       9,
	"kProtectionDomain": 10,
	"kVolumeGroup":      20,
	"kVirtualNic":       15,
}

var MapToLowerCase = map[string]string{
	"kDisk":             "disk",
	"kVDisk":            "disk",
	"kVM":               "vm",
	"kContainer":        "container",
	"kStoragePool":      "storage_pool",
	"kNode":             "node",
	"kCluster":          "cluster",
	"kVirtualDisk":      "virtual_disk",
	"kRemoteSite":       "remote_site",
	"kProtectionDomain": "protection_domain",
	"kVolumeGroup":      "volume_group",
	"kVirtualNic":       "virtual_nic",
}

var RANDOM_NUMBERS = generateRandomNumbers()

var arSvc *arithmos_client.Arithmos

var AGENT_STATS = []string{"kVM", "kVirtualDisk", "kVirtualNic"}

var MASTER_STATS = []string{"kContainer"}

var UNWANTED_SERVICES = []string{
	"Janus", "Aequitas", "Anduril",
	"Flow", "NutanixGuestTools", "XTrim",
	"ClusterHealth", "Delphi", "Lazan",
	"Catalog",
}

var SETUP_FLAGS = []map[string]string{
	{
		"flag":  "arithmos_rss_mem_limit",
		"value": "2147483648",
		"file":  "~/config/genesis.gflags",
	},
	{
		"flag":  "stargate_max_system_memory_reservation_gb",
		"value": "1",
		"file":  "~/config/stargate.gflags",
	},
	{
		"flag":  "insights_rss_share_mb",
		"value": "1500",
		"file":  "~/config/insights_server.gflags",
	},
	{
		"flag":  "arithmos_data_sender_max_entity_fetch_count",
		"value": "20000",
		"file":  "~/config/insights_uploader.gflags",
	},
	{
		"flag":  "pe_tomcat_heap_size_mb",
		"value": "2048",
		"file":  "~/config/genesis.gflags",
	},
}

var COMMANDS = map[string]string{
	"get_disabled_services": "zkcat /appliance/logical/cluster_disabled_services",
	"service_disable":       "echo '%s'| zkwrite /appliance/logical/cluster_disabled_services",
	"file_exists":           "(ls %s >> /dev/null 2>&1 && echo yes) || echo no",
	"read_file":             "cat %s",
	"write_file":            "echo '%s' >> %s",
	"restart_cluster":       "genesis stop all && genesis start && cluster start",
	"clean_arithmos_logs":   "rm -f /home/nutanix/data/logs/arithmos*",
	"create_vm":             "acli vm.create %s num_vcpus=%s memory=%s",
	"clone_vm":              "acli vm.clone %s clone_from_vm=%s",
	"delete_vm":             "echo yes | acli vm.delete %s ",
	"on_vm":                 "acli vm.on %s",
	"off_vm":                "acli vm.off %s",
	"power_on":              "acli vm.on %s",
	"create_disk":           "acli vm.disk_create %s create_size=%s container=SelfServiceContainer",
	"list_net":              "acli -o json net.list",
	"create_net":            "acli net.create vlan0 vlan=0",
	"create_nic":            "acli vm.nic_create %s network=vlan0",
	"is_process_alive":      "ps -p %s",
	"check_alert_gen":       "ls /home/nutanix/bin/alert_generator.py",
	"get_current_processes": "ps -aux | grep -e virtual_pe.go -e alert_generator.py | grep -v grep",
	"kill_process":          "kill %s",
	"ctr_create":            "~/prism/cli/ncli container create sp-name=%s name=%s",
	"sp_list":               "~/prism/cli/ncli sp list | grep Name",
}

type ArithmosCurrentStatGenerator struct {
	svmips             []string
	genericStatList    map[int][]string
	syncedStats        map[string][]string
	genericSyncedStats map[string][]string
	entityIDs          map[string][]string
	baseCtrName        string
	vmNamePrefix       string
	arSvc              *arithmos_client.Arithmos
	insights           *insights_interface.InsightsService
}

// Fixed getEntityIDsFromInsights function that properly handles virtual NIC entity IDs
func (a *ArithmosCurrentStatGenerator) getEntityIDsFromInsights(entityType string, vmEntityGUIDs []string) ([]string, error) {
	if a.insights == nil {
		return nil, fmt.Errorf("insights service not initialized")
	}

	logf(LOG_DEBUG, "Querying insights for %s entities with %d VM entity GUIDs", entityType, len(vmEntityGUIDs))

	// Build the query to get entities of the specified type that belong to the given VMs
	filterString := a.buildVMFilterString(vmEntityGUIDs)
	logf(LOG_DEBUG, "Building insights query for %s with filter: %s", entityType, filterString)
	
	query := fmt.Sprintf(`
		query {
			entity_list {
				entity_type_name: "%s"
			}
			where_clause {
				comparison_expr {
					lhs {
						leaf {
							column: "vm"
						}
					}
					operator: kIN
					rhs {
						leaf {
							value {
								str_list {
									%s
								}
							}
						}
					}
				}
			}
			group_by {
				raw_columns {
					column: "_arithmos_id_"
				}
			}
			query_name: "where_clause"
		}
	`, entityType, filterString)
	
	logf(LOG_DEBUG, "Generated insights query: %s", query)

	arg := &insights_interface.GetEntitiesWithMetricsArg{}
	err := proto.UnmarshalText(query, arg)
	if err != nil {
		logf(LOG_ERROR, "Failed to unmarshal %s query: %v", entityType, err)
		logf(LOG_ERROR, "Query was: %s", query)
		return nil, fmt.Errorf("failed to unmarshal query: %v", err)
	}

	response := &insights_interface.GetEntitiesWithMetricsRet{}
	err = a.insights.SendMsgWithTimeout("GetEntitiesWithMetrics", arg, response, nil, 60)
	if err != nil {
		return nil, fmt.Errorf("failed to query insights: %v", err)
	}
	
	logf(LOG_DEBUG, "%s insights response: %+v", entityType, response)

	var entityIDs []string
	logf(LOG_DEBUG, "Insights query completed for %s, processing results", entityType)
	logf(LOG_DEBUG, "Response has %d group results", len(response.GroupResultsList))
	
	if len(response.GroupResultsList) > 0 {
		groupResult := response.GroupResultsList[0]
		logf(LOG_DEBUG, "First group has %d raw results", len(groupResult.RawResults))
		
		for i, result := range groupResult.RawResults {
			logf(LOG_DEBUG, "Processing result %d: %+v", i, result)
			
			if len(result.MetricDataList) > 0 {
				metricData := result.MetricDataList[0]
				logf(LOG_DEBUG, "Metric data has %d value lists", len(metricData.ValueList))
				
				if len(metricData.ValueList) > 0 {
					valueList := metricData.ValueList[0]
					logf(LOG_DEBUG, "Value list: %+v", valueList)
					
					arithmosID := valueList.Value.GetStrValue()
					logf(LOG_DEBUG, "Extracted arithmos ID: %s", arithmosID)
					
					if arithmosID != "" {
						var entityID string
						
						// Parse the arithmos ID format: entity_type::entity_id
						parts := strings.Split(arithmosID, "::")
						logf(LOG_DEBUG, "Split parts: %v", parts)
						
						if len(parts) > 1 {
							entityID = parts[1]
							logf(LOG_DEBUG, "Extracted entity ID: %s", entityID)
						} else {
							entityID = arithmosID
							logf(LOG_DEBUG, "No separator found, using full arithmos ID: %s", entityID)
						}
						
						// FIXED: For virtual_nic, preserve the full entity ID to ensure uniqueness
						if entityType == "virtual_nic" {
							// Keep the full entity ID which should include VM UUID + NIC identifier
							// This ensures virtual NICs have unique IDs different from their parent VMs
							logf(LOG_DEBUG, "Virtual NIC - keeping full entity ID: %s (length: %d)", entityID, len(entityID))
							
							// Only validate length, don't truncate to VM UUID
							if len(entityID) > 255 {
								logf(LOG_WARNING, "Virtual NIC entity ID exceeds 255 chars (%d), truncating: %s", len(entityID), entityID[:50])
								entityID = entityID[:255]
							}
							
							// Ensure the entity ID is not just the VM UUID by checking for additional identifiers
							if !strings.Contains(entityID, ":") && !strings.Contains(entityID, "-") {
								// If it looks like just a VM UUID, append a NIC identifier
								entityID = fmt.Sprintf("%s:vnic_%d", entityID, i)
								logf(LOG_DEBUG, "Virtual NIC - added NIC identifier: %s", entityID)
							}
						}
						
						if entityID != "" {
							logf(LOG_DEBUG, "Adding entity ID: %s", entityID)
							entityIDs = append(entityIDs, entityID)
						} else {
							logf(LOG_WARNING, "Empty entity ID extracted from arithmos ID: %s", arithmosID)
						}
					} else {
						logf(LOG_WARNING, "Empty arithmos ID in result %d", i)
					}
				} else {
					logf(LOG_WARNING, "No value lists in metric data for result %d", i)
				}
			} else {
				logf(LOG_WARNING, "No metric data in result %d", i)
			}
		}
	} else {
		logf(LOG_WARNING, "No group results in insights response")
	}

	logf(LOG_INFO, "Found %d %s entities via insights query", len(entityIDs), entityType)
	
	if len(entityIDs) > 0 {
		maxLog := 5
		if len(entityIDs) < maxLog {
			maxLog = len(entityIDs)
		}
		logf(LOG_DEBUG, "Sample %s entity IDs: %v", entityType, entityIDs[:maxLog])
	}
	
	return entityIDs, nil
}

func (a *ArithmosCurrentStatGenerator) buildVMFilterString(vmEntityGUIDs []string) string {
	var filterParts []string
	for _, entityGUID := range vmEntityGUIDs {
		if strings.TrimSpace(entityGUID) != "" {
			filterParts = append(filterParts, fmt.Sprintf("value_list: \"%s\"", entityGUID))
		}
	}
	
	filterString := strings.Join(filterParts, "\n\t\t\t\t\t\t\t\t")
	logf(LOG_DEBUG, "Built filter string with %d parts: %s", len(filterParts), filterString)
	return filterString
}

// Enhanced fallback function that generates unique virtual NIC entity IDs
func (a *ArithmosCurrentStatGenerator) generateFallbackEntityIDs(entityType string, vmIDs []string) []string {
	entityIDs := make([]string, 0)
	
	var count int
	if entityType == "virtual_disk" {
		count = diskCount
	} else if entityType == "virtual_nic" {
		count = vnicCount
	}
	
	for _, vmID := range vmIDs {
		if vmID == "" {
			continue
		}
		
		for i := 0; i < count; i++ {
			if entityType == "virtual_nic" {
				// Generate unique virtual NIC IDs that are different from VM IDs
				// Include both VM UUID and NIC index to ensure uniqueness
				entityID := fmt.Sprintf("%s:vnic_%d", vmID, i)
				entityIDs = append(entityIDs, entityID)
				logf(LOG_DEBUG, "Generated fallback virtual NIC ID: %s", entityID)
			} else {
				entityIDs = append(entityIDs, fmt.Sprintf("%s_%s_%d", vmID, entityType, i))
			}
		}
	}
	
	logf(LOG_INFO, "Generated %d fallback %s entity IDs for %d VMs", len(entityIDs), entityType, len(vmIDs))
	return entityIDs
}

func logf(level int, format string, args ...interface{}) {
	if level <= logLevel || level == LOG_VM {
		var prefix string
		switch level {
		case LOG_ERROR:
			prefix = "[ERROR] "
		case LOG_WARNING:
			prefix = "[WARN]  "
		case LOG_INFO:
			prefix = "[INFO]  "
		case LOG_DEBUG:
			prefix = "[DEBUG] "
		case LOG_TRACE:
			prefix = "[TRACE] "
		case LOG_VM:
			prefix = "[VM]    "
		}
		log.Printf(prefix+format, args...)
	}
}

func generateRandomNumbers() []int64 {
	var numbers []int64
	for i := int64(10000); i <= 10100; i += 1000 {
		numbers = append(numbers, i)
	}
	return numbers
}