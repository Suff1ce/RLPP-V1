import numpy as np

def find_behavior_index(EVT01, EVT02, EVT03, EVT04, EVT05, EVT06, EVT07, EVT08):
    # Find the behavior time index from the original events from Plexon
    # CAUTION BEFORE USE. THE EVENT DEFINE MAY CHANGE

    # Remove possible repeated records
    EVT01 = np.unique(EVT01[:,0])
    EVT02 = np.unique(EVT02[:,0])
    EVT03 = np.unique(EVT03[:,0])
    EVT04 = np.unique(EVT04[:,0])
    EVT05 = np.unique(EVT05[:,0])
    EVT06 = np.unique(EVT06[:,0])
    EVT07 = np.unique(EVT07[:,0])
    EVT08 = np.unique(EVT08[:,0])

    # Detect press events
    INDEX_press = []
    # All high press
    for evt in EVT01:
        idx = np.where((evt > (EVT06 - 0.001)) & (evt < (EVT06 + 0.001)))[0]
        INDEX_press.extend(idx)
    
    # All low press
    for evt in EVT02:
        idx = np.where((evt > (EVT06 - 0.001)) & (evt < (EVT06 + 0.001)))[0]
        INDEX_press.extend(idx)

    press = EVT06[INDEX_press]

    # Detect release events based on press
    release = []
    for p in press:  # Two types of release: early release(04) & release(07)
        a = np.where(EVT04 > p)[0]
        b = np.where(EVT07 > p)[0]
        if len(a) == 0 and len(b) > 0:
            release.append(EVT07[b[0]])
        elif len(b) == 0 and len(a) > 0:
            release.append(EVT04[a[0]])
        elif len(a) > 0 and len(b) > 0:
            release.append(min(EVT04[a[0]], EVT07[b[0]]))
        else:
            print("Warning: Unable to find release event!")
    
    # Put press-release together for return
    press_release = np.array([press, release]).T

    # Delete early release events
    ind_early = []
    for evt in EVT04:
        idx = np.where(EVT06 < evt)[0]
        if len(idx) > 0:
            ind_early.append(idx[-1])
    
    EVT06 = np.delete(EVT06, ind_early)

    # Detect high success events
    INDEX = []
    for evt in EVT01:
        idx = np.where((evt > (EVT03 - 0.001)) & (evt <= (EVT03 + 0.001)))[0]
        if len(idx) > 0:
            INDEX.append(idx[-1])
    
    high_success = EVT03[INDEX]

    # Detect low success events
    INDEX2 = []
    for evt in EVT02:
        idx = np.where((evt > (EVT03 - 0.001)) & (evt <= (EVT03 + 0.001)))[0]
        if len(idx) > 0:
            INDEX2.append(idx[-1])

    low_success = EVT03[INDEX2]

    # Detect success high press
    INDEX3 = []
    for hs in high_success:
        idx = np.where(EVT06 < hs)[0]
        if len(idx) > 0:
            INDEX3.append(idx[-1])
    
    success_high_press = EVT06[INDEX3]

    # Detect success low press
    INDEX4 = []
    for ls in low_success:
        idx = np.where(EVT06 < ls)[0]
        if len(idx) > 0:
            INDEX4.append(idx[-1])
    
    success_low_press = EVT06[INDEX4]

    # Detect success high start
    INDEX5 = []
    for hs in high_success:
        idx = np.where(EVT05 < hs)[0]
        if len(idx) > 0:
            INDEX5.append(idx[-1])
    
    success_high_start = EVT05[INDEX5]

    # Detect success low start
    INDEX6 = []
    for ls in low_success:
        idx = np.where(EVT05 < ls)[0]
        if len(idx) > 0:
            INDEX6.append(idx[-1])
    
    success_low_start = EVT05[INDEX6]

    # Detect success high release
    INDEX7 = []
    for hs in high_success:
        idx = np.where(EVT07 > hs)[0]
        if len(idx) > 0:
            INDEX7.append(idx[0])
    
    success_high_release = EVT07[INDEX7]

    # Detect success low release
    INDEX8 = []
    for ls in low_success:
        idx = np.where(EVT07 > ls)[0]
        if len(idx) > 0:
            INDEX8.append(idx[0])
    
    success_low_release = EVT07[INDEX8]

    # Put them together for return
    success_start = np.sort(np.concatenate([success_high_start, success_low_start]))
    rest = np.column_stack([success_start - 0.5, success_start])
    press_high = np.column_stack([success_high_press, success_high_release])
    press_low = np.column_stack([success_low_press, success_low_release])

    return rest, press_low, press_high, press_release
