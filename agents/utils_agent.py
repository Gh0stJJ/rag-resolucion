#utils_agent.py
from datetime import datetime, timedelta
from calendar import monthrange

def get_current_year_range():
    """Calcula y devuelve el primer y último día del año actual."""
    anio_actual = datetime.now().year
    fecha_inicio = f"2021-01-01"
    fecha_fin = f"{anio_actual}-12-31"
    return fecha_inicio, fecha_fin

def get_month_range(year: int, month: int):
    """Calcula el primer y último día de un mes y año específicos."""
    _, last_day = monthrange(year, month)
    first_day = 1
    fecha_inicio = f"{year}-{month:02d}-{first_day:02d}"
    fecha_fin = f"{year}-{month:02d}-{last_day:02d}"
    return fecha_inicio, fecha_fin

def get_today():
    """Devuelve la fecha actual en formato YYYY-MM-DD."""
    return datetime.now().strftime('%Y-%m-%d')

def get_yesterday():
    """Devuelve la fecha de ayer en formato YYYY-MM-DD."""
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')
