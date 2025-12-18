import trackpy as tp

def track_with_trackpy(detections_df, search_range):
    tracked = tp.link_df(detections_df, search_range=search_range)
    return tracked
