# dynamic tool: calculate_pga_esteva_rosenblueth
def calculate_pga_esteva_rosenblueth(magnitude: float, distance: float, c: float = 0.001, alpha: float = 1.5, beta: float = 1.2) -> dict:
    """
    使用 Esteva 和 Rosenblueth (1964) 的 PGA 经验公式计算峰值地面加速度。
    公式：a = c * exp(α*M) * R^(-β)
    参数：
        magnitude (float): 地震震级 M
        distance (float): 震中距 R（单位：km）
        c (float): 模型常数（默认 0.001，可根据区域调整）
        alpha (float): 震级衰减系数（默认 1.5）
        beta (float): 距离衰减指数（默认 1.2）
    返回：
        dict: 包含 status, pga (单位：g), 和参数说明
    """
    import math
    if magnitude <= 0:
        return {'status': 'error', 'reason': '震级必须大于0'}
    if distance <= 0:
        return {'status': 'error', 'reason': '震中距必须大于0'}
    try:
        pga = c * math.exp(alpha * magnitude) * (distance ** (-beta))
        return {
            'status': 'success',
            'pga': pga,
            'units': 'g',
            'parameters': {
                'magnitude': magnitude,
                'distance_km': distance,
                'c': c,
                'alpha': alpha,
                'beta': beta
            },
            'formula': 'a = c * exp(α*M) * R^(-β)',
            'reference': 'Esteva and Rosenblueth (1964)'
        }
    except Exception as e:
        return {'status': 'error', 'reason': f'计算失败: {type(e).__name__}'}
