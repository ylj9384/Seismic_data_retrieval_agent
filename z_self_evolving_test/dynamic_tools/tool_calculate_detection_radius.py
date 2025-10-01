# dynamic tool: calculate_detection_radius
def calculate_detection_radius(depth: float, magnitude: float):
    """
    根据经验公式估算地震在地表的影响半径（单位：km）
    参数:
        depth: 地震深度（km）
        magnitude: 地震震级
    返回: 包含地表影响半径的字典
    """
    # 经验公式：log(R) = 0.12*M + 0.41*log(depth + 10) + 0.54
    try:
        if magnitude <= 0 or depth <= 0:
            return {'status': 'error', 'reason': '震级和深度必须为正数'}
        import math
        log_R = 0.12 * magnitude + 0.41 * math.log10(depth + 10) + 0.54
        radius = 10 ** log_R
        return {
            'status': 'success',
            'depth_km': depth,
            'magnitude': magnitude,
            'detection_radius_km': radius
        }
    except Exception as e:
        return {'status': 'error', 'reason': f'{type(e).__name__}: {e}'}
