# Risk levels and mapping from Section 9
RISK_MAP = {
    'Pneumothorax'     : 'CRITICAL',
    'Pneumonia'        : 'CRITICAL',
    'Edema'            : 'CRITICAL',
    'Effusion'         : 'HIGH',
    'Consolidation'    : 'HIGH',
    'Cardiomegaly'     : 'HIGH',
    'Atelectasis'      : 'HIGH',
    'Mass'             : 'HIGH',
    'Infiltration'     : 'MEDIUM',
    'Nodule'           : 'MEDIUM',
    'Emphysema'        : 'MEDIUM',
    'Fibrosis'         : 'MEDIUM',
    'Pleural_Thickening': 'LOW',
    'Hernia'           : 'LOW',
    'No Finding'       : 'NORMAL',
}

RISK_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']

def get_risk_level(positive_conditions):
    """Determine highest risk level among positive findings."""
    if not positive_conditions:
        return 'UNCERTAIN'
    
    # Exclude "No Finding" for risk scoring if other things exist
    disease_conditions = [c for c in positive_conditions if c != 'No Finding']
    if not disease_conditions:
        return 'NORMAL'
        
    risks = [RISK_MAP.get(c, 'LOW') for c in disease_conditions]
    
    # Return the most severe risk level found
    for level in RISK_ORDER:
        if level in risks:
            return level
            
    return 'UNCERTAIN'

def get_case_flags(positive_conditions, kept_boxes, overlap_classes):
    """Generate diagnostic case flags based on Section 12."""
    flags = []
    
    # Case 5: Conflicting No Finding
    if "No Finding" in positive_conditions and len(positive_conditions) > 1:
        flags.append("conflicting_no_finding")
        
    # Case 6: Nothing above threshold
    if not positive_conditions:
        flags.append("low_confidence_all")
        
    for cond in positive_conditions:
        if cond == "No Finding":
            continue
            
        # Normalize for comparison
        norm_cond = cond.replace('_', ' ')
        if norm_cond not in overlap_classes:
            flags.append(f"heatmap_only:{cond}")
            
    return flags
