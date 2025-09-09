# VirtualNic Entity ID Fix

## Problem
The original code was truncating VirtualNic entity IDs to just the VM UUID, making all VirtualNic entities for the same VM have identical IDs. This breaks Arithmos stats collection because each entity needs a unique identifier.

## Solution
The fix involves three key changes:

### 1. Import crypto/sha256 package
Add to your imports:
```go
import (
    "crypto/sha256"
    // ... other imports
)
```

### 2. Fix getEntityIDsFromInsights function
Replace the virtual_nic handling section with:

```go
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
}
```

### 3. Update generateFallbackEntityIDs function
Replace the fallback generation logic with:

```go
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
```

### 4. Add validation function (optional but recommended)
Add this helper function to validate uniqueness:

```go
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
```

### 5. Use validation in getEntityIDs function
Add validation after collecting entity IDs:

```go
// After collecting entity IDs for each type, validate uniqueness
for entityType, ids := range a.entityIDs {
    if len(ids) > 0 {
        a.validateEntityIDUniqueness(entityType, ids)
    }
}
```

## Key Changes Summary
1. **Preserve uniqueness**: Don't truncate VirtualNic IDs to just VM UUIDs
2. **Handle long IDs**: Use SHA256 hashing for IDs that exceed 255 characters
3. **Maintain readability**: Keep VM UUID prefix when possible
4. **Fallback generation**: Create unique fallback IDs for VirtualNics
5. **Add validation**: Ensure no duplicate entity IDs within each type

This fix ensures that each VirtualNic entity has a unique identifier while staying within Arithmos length limits, allowing proper stats collection for all VirtualNic entities.