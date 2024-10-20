#Fernando José Figueroa Olivo 3103131
#Pamela Beatriz Gómez Rosales 3103132
#Diego José Viana Landaverde 3103134

class Enfermedad:
    def __init__(self, nombre, sintomas):
        self.nombre = nombre
        self.sintomas = sintomas

    def __str__(self):
        sintomas_str = ', '.join([str(sintoma) for sintoma in self.sintomas])
        return f'{self.nombre}: {sintomas_str}'

class Sintoma:
    def __init__(self, nombre):
        self.nombre = nombre

    def __str__(self):
        return self.nombre

    def __eq__(self, other):
        if isinstance(other, Sintoma):
            return self.nombre == other.nombre
        return False

    def __hash__(self):
        return hash(self.nombre)

# Crear un diccionario que mapee cada síntoma a las enfermedades que lo contienen
def crear_grafo_y_diccionario_sintomas(enfermedades):
    grafo = {}
    sintoma_a_enfermedades = {}
    for enfermedad in enfermedades:
        for sintoma in enfermedad.sintomas:
            if sintoma not in grafo:
                grafo[sintoma] = set()
            for otro_sintoma in enfermedad.sintomas:
                if sintoma != otro_sintoma:
                    grafo[sintoma].add(otro_sintoma)
            if sintoma not in sintoma_a_enfermedades:
                sintoma_a_enfermedades[sintoma] = []
            sintoma_a_enfermedades[sintoma].append(enfermedad)
    print(" ")
    print("Grafo de síntomas:", {str(k): {str(v) for v in vals} for k, vals in grafo.items()})
    print(" ")
    print("Diccionario de síntomas a enfermedades:", {str(k): [str(v) for v in vals] for k, vals in sintoma_a_enfermedades.items()})
    print(" ")
    return grafo, sintoma_a_enfermedades

# Pedir al usuario que ingrese los síntomas
def obtener_sintomas_usuario():
    sintomas_usuario = input("Por favor, escriba los síntomas separados por coma: ").split(',')
    sintomas_usuario = [Sintoma(sintoma.strip()) for sintoma in sintomas_usuario]
    print(" ")
    print("Síntomas ingresados por el usuario:", [str(sintoma) for sintoma in sintomas_usuario])
    print(" ")
    return sintomas_usuario

# Realizar la búsqueda hacia atrás
def buscar_enfermedades_parcial(sintomas_usuario, sintoma_a_enfermedades):
    enfermedad_coincidencias = {}
    for sintoma_usuario in sintomas_usuario:
        print(" ")
        print(f"Procesando síntoma del usuario: {sintoma_usuario}")
        if sintoma_usuario in sintoma_a_enfermedades:
            for enfermedad in sintoma_a_enfermedades[sintoma_usuario]:
                if enfermedad not in enfermedad_coincidencias:
                    enfermedad_coincidencias[enfermedad] = 0
                enfermedad_coincidencias[enfermedad] += 1
            print(f"Enfermedades posibles después de procesar {sintoma_usuario}: {[str(enf) for enf in sintoma_a_enfermedades[sintoma_usuario]]}")
            print(" ")
        else:
            print(f"Síntoma {sintoma_usuario} no encontrado en el diccionario.")
            print(" ")
    return enfermedad_coincidencias

# Ejemplo de uso
enfermedades = [
    Enfermedad('Gripe', [Sintoma('Fiebre'), Sintoma('Dolor de estomago'), Sintoma('Dolor de espalda')]),
    Enfermedad('Resfriado', [Sintoma('Dolor de espalda'), Sintoma('Dolor de cabeza'), Sintoma('Tos')]),
    Enfermedad('COVID-19', [Sintoma('Dolor de estomago'), Sintoma('Tos'), Sintoma('Dolor de espalda')]),
    Enfermedad('Alergia', [Sintoma('Estornudos'), Sintoma('Picazón en los ojos'), Sintoma('Congestión nasal')]),
    Enfermedad('Migraña', [Sintoma('Dolor de cabeza'), Sintoma('Náuseas'), Sintoma('Sensibilidad a la luz')]),
    Enfermedad('Gastroenteritis', [Sintoma('Dolor de estomago'), Sintoma('Diarrea'), Sintoma('Vómitos')]),
    Enfermedad('Amigdalitis', [Sintoma('Dolor de garganta'), Sintoma('Fiebre'), Sintoma('Dificultad para tragar')]),
    Enfermedad('Bronquitis', [Sintoma('Tos'), Sintoma('Flema'), Sintoma('Dificultad para respirar')]),
    Enfermedad('Sinusitis', [Sintoma('Dolor facial'), Sintoma('Congestión nasal'), Sintoma('Dolor de cabeza')]),
    Enfermedad('Otitis', [Sintoma('Dolor de oído'), Sintoma('Fiebre'), Sintoma('Pérdida de audición')])
]

grafo, sintoma_a_enfermedades = crear_grafo_y_diccionario_sintomas(enfermedades)
sintomas_usuario = obtener_sintomas_usuario()
posibles_enfermedades = buscar_enfermedades_parcial(sintomas_usuario, sintoma_a_enfermedades)

# Mostrar el resultado
if posibles_enfermedades:
    enfermedad_coincidencias = buscar_enfermedades_parcial(sintomas_usuario, sintoma_a_enfermedades)
    if enfermedad_coincidencias:
        enfermedad_mas_probable = max(enfermedad_coincidencias, key=enfermedad_coincidencias.get)
        print("La enfermedad más probable que coincide con los síntomas ingresados es:")
        print(enfermedad_mas_probable)
    else:
        print("No se encontraron enfermedades que coincidan con los síntomas ingresados.")
else:
    print("No se encontraron enfermedades que coincidan con los síntomas ingresados.")