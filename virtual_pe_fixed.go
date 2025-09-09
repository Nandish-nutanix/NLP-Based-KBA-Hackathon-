// Fixed getEntityIDsFromInsights function with proper VirtualNic entity ID handling
package main

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"log"
)

// getEntityIDsFromInsights queries the insights interface to get entity IDs for virtual disks and virtual NICs
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
						
						if entityType == "virtual_nic" {
							// For VirtualNic, preserve uniqueness while keeping within length limits
							// The full ID format is typically: vm_uuid:mac_address or similar
							parts := strings.Split(arithmosID, "::")
							logf(LOG_DEBUG, "Virtual NIC - splitting by '::': %v", parts)
							
							if len(parts) > 1 {
								entityID = parts[1]
								logf(LOG_DEBUG, "Virtual NIC - extracted entity ID: %s", entityID)
								
								// Keep the full unique identifier but ensure it's within limits
								// Don't truncate to just VM UUID as that removes uniqueness
								if len(entityID) > 255 {
									// If too long, create a hash-based unique identifier
									hasher := sha256.New()
									hasher.Write([]byte(entityID))
									hashBytes := hasher.Sum(nil)
									// Use first part of original ID + hash to maintain some readability
									colonIndex := strings.Index(entityID, ":")
									if colonIndex > 0 && colonIndex < 100 {
										prefix := entityID[:colonIndex]
										entityID = fmt.Sprintf("%s_%x", prefix, hashBytes[:8])
									} else {
										entityID = fmt.Sprintf("vnic_%x", hashBytes[:16])
									}
									logf(LOG_DEBUG, "Virtual NIC - created hash-based ID: %s (length: %d)", entityID, len(entityID))
								}
							} else {
								// No separator found, use the full ID but ensure uniqueness
								if len(arithmosID) > 255 {
									hasher := sha256.New()
									hasher.Write([]byte(arithmosID))
									hashBytes := hasher.Sum(nil)
									entityID = fmt.Sprintf("vnic_%x", hashBytes[:16])
									logf(LOG_DEBUG, "Virtual NIC - created hash-based ID from full arithmos ID: %s", entityID)
								} else {
									entityID = arithmosID
									logf(LOG_DEBUG, "Virtual NIC - using full arithmos ID: %s", entityID)
								}
							}
						} else {
							// For other entity types (virtual_disk, etc.), use standard extraction
							parts := strings.Split(arithmosID, "::")
							logf(LOG_DEBUG, "Split parts: %v", parts)
							
							if len(parts) > 1 {
								entityID = parts[1]
								logf(LOG_DEBUG, "Extracted entity ID: %s", entityID)
							} else {
								entityID = arithmosID
								logf(LOG_DEBUG, "No separator found, using full arithmos ID: %s", entityID)
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

// Additional helper function to validate entity ID uniqueness
func (a *ArithmosCurrentStatGenerator) validateEntityIDUniqueness(entityType string, entityIDs []string) bool {
	seen := make(map[string]bool)
	duplicates := []string{}
	
	for _, id := range entityIDs {
		if seen[id] {
			duplicates = append(duplicates, id)
		} else {
			seen[id] = true
		}
	}
	
	if len(duplicates) > 0 {
		logf(LOG_ERROR, "Found duplicate entity IDs for %s: %v", entityType, duplicates)
		return false
	}
	
	logf(LOG_DEBUG, "All %d %s entity IDs are unique", len(entityIDs), entityType)
	return true
}

// Updated generateFallbackEntityIDs to ensure unique VirtualNic IDs
func (a *ArithmosCurrentStatGenerator) generateFallbackEntityIDs(entityType string, vmIDs []string) []string {
	entityIDs := make([]string, 0)
	
	for _, vmID := range vmIDs {
		if vmID == "" {
			continue
		}
	}
	
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
				// Generate unique VirtualNic IDs with MAC-like suffix for uniqueness
				entityIDs = append(entityIDs, fmt.Sprintf("%s:vnic%02d", vmID, i))
			} else {
				entityIDs = append(entityIDs, fmt.Sprintf("%s_%s_%d", vmID, entityType, i))
			}
		}
	}
	
	logf(LOG_INFO, "Generated %d fallback %s entity IDs for %d VMs", len(entityIDs), entityType, len(vmIDs))
	return entityIDs
}