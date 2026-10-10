## Uso de IA para aprendizaje de Git y GitHub (Gemini y Copilot)
### 1. Archivos de data/raw eran agregados a Git pese a estar en .gitignore

***Prompt:***
 Se consultó por qué los archivos ubicados en data/raw seguían siendo incluidos al ejecutar git add --all, a pesar de que la ruta había sido agregada al archivo .gitignore.

***Respuesta:***
 Se explicó que el problema podía deberse a dos causas: una regla incorrecta en el .gitignore o que los archivos ya estuvieran siendo rastreados por Git antes de crear la regla de exclusión.

Aprendizaje obtenido:
 La persona aprendió que .gitignore solo evita que Git agregue archivos nuevos, pero no elimina del control de versiones aquellos que ya fueron agregados previamente.

### 2. Caso de Consulta: Solución a Problemas de Edición en Archivos Markdown en PyChar

***Prompt:***
"por que no puedo editar, agregar texto al ese archivo recién creado?"

***Respuesta*** (resumida)

Se identificaron tres causas principales por las cuales PyCharm impide la edición de un archivo .md:

Archivo en modo de solo lectura (Read-Only): Bloqueo de permisos a nivel de sistema o dentro de la interfaz del IDE (icono de candado).

Modo de vista incorrecto: El archivo se visualizaba únicamente en la pestaña de Preview (Vista Previa) en lugar del modo Editor o Split.

Pérdida de foco o indexación: Bloqueo temporal de la interfaz por procesos en segundo plano o selección de panel incorrecto.

Aprendizaje Obtenido

Se comprendió el comportamiento de los entornos de desarrollo (IDE) como PyCharm al gestionar archivos de documentación en formato Markdown. 