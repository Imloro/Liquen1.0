import pandas as pd
import numpy as np
from scipy.optimize import linprog
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OrderOptimizer:
    def __init__(self):
        self.vehiculos = {
            'Furgoneta1': 3.3,
            'Furgoneta2': 4.3,
            'Furgoneta3': 6.6,
            'Furgoneta4': 8.6
        }
        self.distancias = {
            ('A', '1'): 21, ('B', '1'): 20, ('C', '1'): 12,
            ('D', '1'): 20, ('E', '1'): 8,
            ('A', '2'): 21, ('B', '2'): 22, ('C', '2'): 18,
            ('D', '2'): 17, ('E', '2'): 13,
            ('A', '3'): 18, ('B', '3'): 23, ('C', '3'): 22,
            ('D', '3'): 20, ('E', '3'): 19
        }

    def cargar_datos(self, ruta: str) -> pd.DataFrame:
        """Carga y limpia los datos"""
        try:
            df = pd.read_excel(ruta)
            
            # Validación y limpieza
            required_cols = ['Almacén_Origen', 'Zona_Destino', 'Volumen_m³', 'Urgencia', 'Margen']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Faltan columnas requeridas: {required_cols}")
            
            # Limpieza de datos
            df = df.dropna(subset=required_cols)
            df['Zona_Destino'] = df['Zona_Destino'].astype(str)
            
            # Asegurar tipos numéricos
            numeric_cols = ['Urgencia', 'Margen', 'Volumen_m³']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}")
            return pd.DataFrame()

    def priorizar_pedidos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula puntuación evitando divisiones por cero"""
        try:
            df = df.copy()
            
            # Manejar columnas opcionales
            optional_columns = {
                'Service_Profitability': 0,
                'Volume': 0,
                'Product_Type': 0
            }
            for col, default_val in optional_columns.items():
                if col not in df.columns:
                    df[col] = default_val  # Agrega como columna de ceros
            
            # Función de normalización segura
            def safe_scale(s: pd.Series, invert: bool = False) -> pd.Series:
                s = pd.to_numeric(s, errors='coerce').fillna(0)
                max_val = s.max() or 1  # Evita división por cero
                return (1 - (s / max_val)) if invert else (s / max_val)
            
            # Cálculo de la puntuación
            df['Puntuacion_Total'] = (
                0.25 * safe_scale(df['Urgencia']) +
                0.20 * safe_scale(df['Margen']) +
                0.15 * safe_scale(df['Service_Profitability']) +
                0.10 * safe_scale(df['Volume'], invert=True) +
                0.10 * safe_scale(df['Product_Type'])
            )
            
            return df.sort_values('Puntuacion_Total', ascending=False)
            
        except Exception as e:
            logging.error(f"Error en priorización: {str(e)}")
            return df

    def optimizar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimización con validación de valores numéricos"""
        try:
            if df.empty:
                return pd.DataFrame()
                
            # Preparación de datos
            df = df.reset_index(drop=True)
            vehiculos = list(self.vehiculos.keys())
            
            # Vector de costos con protección contra NaN/inf
            c = []
            for vehiculo in vehiculos:
                for _, row in df.iterrows():
                    distancia = self.distancias.get(
                        (row['Almacén_Origen'], row['Zona_Destino']), 50
                    )
                    puntuacion = max(row['Puntuacion_Total'], 0.001)  # Evita división por cero
                    costo = distancia / puntuacion
                    c.append(costo)
            
            # Convertir a array numpy y limpiar valores inválidos
            c = np.nan_to_num(np.array(c), nan=1e6, posinf=1e6, neginf=1e6)
            
            # Restricciones
            A_ub, b_ub, A_eq, b_eq = self._construir_restricciones(df)
            
            # Optimización
            res = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=(0, 1),
                method='highs'
            )
            
            return self._procesar_resultados(res, df, vehiculos) if res.success else pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Error en optimización: {str(e)}")
            return pd.DataFrame()

    def _construir_restricciones(self, df: pd.DataFrame) -> tuple:
        """Construye matrices de restricciones"""
        num_pedidos = len(df)
        num_vehiculos = len(self.vehiculos)
        
        # Restricciones de capacidad
        A_ub = np.zeros((num_vehiculos, num_pedidos * num_vehiculos))
        b_ub = []
        for j, cap in enumerate(self.vehiculos.values()):
            A_ub[j, j*num_pedidos:(j+1)*num_pedidos] = df['Volumen_m³'].values
            b_ub.append(cap)
        
        # Restricciones de asignación completa
        A_eq = np.zeros((num_pedidos, num_pedidos * num_vehiculos))
        for i in range(num_pedidos):
            A_eq[i, i::num_pedidos] = 1
        b_eq = np.ones(num_pedidos)
        
        return A_ub, b_ub, A_eq, b_eq

    def _procesar_resultados(self, res, df, vehiculos) -> pd.DataFrame:
        """Convierte resultados en DataFrame"""
        asignaciones = []
        x = res.x
        num_pedidos = len(df)
        
        for j, vehiculo in enumerate(vehiculos):
            for i in range(num_pedidos):
                if x[j*num_pedidos + i] > 0.01:  # Umbral de asignación
                    row = df.iloc[i]
                    asignaciones.append({
                        'Pedido_ID': i,
                        'Vehiculo': vehiculo,
                        'Almacen': row['Almacén_Origen'],
                        'Zona': row['Zona_Destino'],
                        'Volumen_Asignado': x[j*num_pedidos + i] * row['Volumen_m³'],
                        'Distancia': self.distancias.get((row['Almacén_Origen'], row['Zona_Destino']), 50),
                        'Puntuacion': row['Puntuacion_Total']
                    })
        
        return pd.DataFrame(asignaciones)

def main():
    optimizer = OrderOptimizer()
    
    # 1. Cargar datos
    df = optimizer.cargar_datos("data_orders.xlsx")
    if df.empty:
        print("No se pudieron cargar los datos")
        return
    
    # 2. Priorizar
    df_priorizado = optimizer.priorizar_pedidos(df)
    print("\nTop 5 pedidos priorizados:")
    print(df_priorizado.head())
    
    # 3. Optimizar
    resultados = optimizer.optimizar(df_priorizado)
    
    if not resultados.empty:
        print("\nResultados de optimización:")
        print(resultados)
        resultados.to_excel("resultados_optimizacion.xlsx", index=False)
    else:
        print("\nNo se encontró solución óptima")
if __name__ == "__main__":
    main()
