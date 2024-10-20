class Enfermedad:
    def __init__(self, nombre, sintomas):
        self.nombre = nombre
        self.sintomas = sintomas
        self.chances = 0

    def agregar_sintoma(self, sintoma):
        self.sintomas.append(sintoma)

    def mostrar_sintomas(self):
        return self.sintomas

    def __str__(self):
        return f'{self.nombre}: {self.sintomas}'

class Sintoma:
    def __init__(self, nombre):
        self.nombre = nombre

    def __str__(self):
        return self.nombre

sintomas = [
    Sintoma('Fiebre'),
    Sintoma('Dolor de estómago'),
    Sintoma('Dolor de espalda'),
    Sintoma('Dolor de cabeza'),
    Sintoma('Tos')
]

enfermedades = [
    Enfermedad('Gripe', [sintomas[0], sintomas[1], sintomas[3]]),
    Enfermedad('Resfriado', [sintomas[2], sintomas[3], sintomas[4]]),
    Enfermedad('COVID-19', [sintomas[1], sintomas[4], sintomas[2]])
]

# Ask user for symptoms

sintomas_usuario = input("Por favor, escriba los síntomas separados por coma: ").split(',')
sintomas_usuario = [Sintoma(sintoma.strip()) for sintoma in sintomas_usuario]

print("Síntomas ingresados:")
for sintoma in sintomas_usuario:
    print(sintoma)

print("\nEnfermedades que coinciden con los síntomas ingresados:")

# Search for coincidences
def check_symptoms(enfermedad, sintomas_usuario):
    coincidencias = 0
    for sintoma in sintomas_usuario:
        if sintoma not in enfermedad.sintomas:
            coincidencias += 1
    return coincidencias

for enfermedad in enfermedades:
    coincidencias = check_symptoms(enfermedad, sintomas_usuario)
    enfermedad.chances = coincidencias

enfermedades.sort(key=lambda x: x.chances, reverse=True)
enfermedades = [enfermedad for enfermedad in enfermedades if enfermedad.chances > 0]

if enfermedades:
    print(f"La enfermedad con más coincidencias es: {enfermedades[0].nombre}")
else:
    print("No se encontraron enfermedades que coincidan con los síntomas ingresados.")
