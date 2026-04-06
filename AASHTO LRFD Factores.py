import math


class PuenteAASHTO:
    """
    Clase principal que define la estructura básica para el cálculo de factores
    de distribución de carga viva según AASHTO LRFD.
    """

    def __init__(self, S, L, ts, Nb, Kg, de):
        self.S = S  # Separación entre vigas (mm)
        self.L = L  # Longitud del vano (mm)
        self.ts = ts  # Espesor de la losa (mm)
        self.Nb = Nb  # Número de vigas
        self.Kg = Kg  # Parámetro de rigidez longitudinal (mm^4)
        self.de = de  # Distancia desde el eje de la viga exterior al borde interior de la barrera (mm)

    def verificar_limites(self):
        """Método a implementar en las clases hijas para verificar rango de aplicabilidad."""
        raise NotImplementedError("Este método debe ser definido en la clase hija.")

    def momento_interior(self, carriles):
        raise NotImplementedError()

    def momento_exterior(self, carriles):
        raise NotImplementedError()

    def cortante_interior(self, carriles):
        raise NotImplementedError()

    def cortante_exterior(self, carriles):
        raise NotImplementedError()

    def regla_de_la_palanca(self):
        """
        La regla de la palanca asume una articulación en la primera viga interior
        y calcula la reacción estática en la viga exterior.
        Para automatizarlo completamente, se requeriría ingresar la geometría
        transversal completa y la posición exacta de los camiones.
        """
        return "Requiere cálculo manual mediante análisis estático (Regla de la Palanca)"


class PuenteVigasTyI(PuenteAASHTO):
    """
    Clase hija para secciones tipo a, e, k (Losa sobre vigas de acero/concreto, Vigas T, T-bulbo).
    """

    def verificar_limites(self):
        """Verifica las limitaciones geométricas de la tabla 4.6.2.2.2b-1 y 4.6.2.2.2d-1"""
        errores = []
        if not (1100 <= self.S <= 4900):
            errores.append(f"Separación S ({self.S} mm) fuera de límite (1100-4900).")
        if not (110 <= self.ts <= 300):
            errores.append(f"Espesor ts ({self.ts} mm) fuera de límite (110-300).")
        if not (6000 <= self.L <= 73000):
            errores.append(f"Longitud L ({self.L} mm) fuera de límite (6000-73000).")
        if self.Nb < 4:
            errores.append(f"Número de vigas Nb ({self.Nb}) debe ser >= 4 para usar estas ecuaciones completas.")
        if not (4e9 <= self.Kg <= 3e12):
            errores.append(f"Parámetro Kg ({self.Kg:e} mm^4) fuera de límite (4x10^9 - 3x10^12).")
        if not (-300 <= self.de <= 1700):
            errores.append(f"Distancia de ({self.de} mm) fuera de límite (-300 a 1700).")

        if errores:
            print("ADVERTENCIA: Geometría fuera del rango de aplicabilidad AASHTO:")
            for error in errores:
                print(" -", error)
        else:
            print("La geometría cumple con los rangos de aplicabilidad de AASHTO.")
        return len(errores) == 0

    # --- FACTORES PARA MOMENTO FLECTOR ---
    def momento_interior(self, carriles):
        """Distribución de momento para vigas interiores."""
        termino_rigidez = (self.Kg / (self.L * (self.ts ** 3))) ** 0.1

        if carriles == 1:
            return 0.06 + ((self.S / 4300) ** 0.4) * ((self.S / self.L) ** 0.3) * termino_rigidez
        elif carriles >= 2:
            return 0.075 + ((self.S / 2900) ** 0.6) * ((self.S / self.L) ** 0.2) * termino_rigidez
        else:
            raise ValueError("Número de carriles inválido.")

    def momento_exterior(self, carriles):
        """Distribución de momento para vigas exteriores."""
        if carriles == 1:
            return self.regla_de_la_palanca()
        elif carriles >= 2:
            e = 0.77 + (self.de / 2800)
            return e * self.momento_interior(carriles)
        else:
            raise ValueError("Número de carriles inválido.")

    # --- FACTORES PARA CORTANTE ---
    def cortante_interior(self, carriles):
        """Distribución de cortante para vigas interiores."""
        if carriles == 1:
            return 0.36 + (self.S / 7600)
        elif carriles >= 2:
            return 0.2 + (self.S / 3600) - ((self.S / 10700) ** 2.0)
        else:
            raise ValueError("Número de carriles inválido.")

    def cortante_exterior(self, carriles):
        """Distribución de cortante para vigas exteriores."""
        if carriles == 1:
            return self.regla_de_la_palanca()
        elif carriles >= 2:
            e = 0.6 + (self.de / 3000)
            return e * self.cortante_interior(carriles)
        else:
            raise ValueError("Número de carriles inválido.")


# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Parámetros de ejemplo para un puente de vigas T
    S = 2500  # 2.5 metros
    L = 18000  # 18 metros
    ts = 200  # 20 cm
    Nb = 5  # 5 vigas
    Kg = 1.5e11  # Calculado previamente en mm^4
    de = 500  # 50 cm

    # Instanciamos la clase hija
    mi_puente = PuenteVigasTyI(S, L, ts, Nb, Kg, de)

    # 1. Verificamos que los datos cumplan los límites
    print("--- Verificación de Límites ---")
    mi_puente.verificar_limites()

    # 2. Calculamos los factores
    print("\n--- Factores de Distribución ---")
    print(f"Momento Interior (2+ carriles): {mi_puente.momento_interior(carriles=2):.3f}")
    print(f"Momento Exterior (2+ carriles): {mi_puente.momento_exterior(carriles=2):.3f}")
    print(f"Cortante Interior (2+ carriles): {mi_puente.cortante_interior(carriles=2):.3f}")
    print(f"Cortante Exterior (2+ carriles): {mi_puente.cortante_exterior(carriles=2):.3f}")