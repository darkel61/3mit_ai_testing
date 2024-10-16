Step by Step


1. Cambiar version de Python a 3.10.2 (Lo puedes hacer con pyenv)
2. Instalar `pip install --upgrade pip setuptools wheel`
3. Instalar los Requirements `pip install -r requirements.txt`
4. Instalar Greykite `pip install greykite`
5. Ejecutar `script.py`


Comentarios:

- Es posible que al momento de instalar los requirements, osqp te de error, simplemente comentalo y ya owo.
- En los entornos de WSL, no vas a tener acceso directo para que el comando `plotly.io.show(fig)` te muestre las graficas, por ende, tendras que usar `fig.write_html("path/to/file.html")` para poder ver la grafica.
- Posiblemente tengas errores con el env de Python, y tengas que usar `python venv` para manejar estas cosas o usar `--break-system-packages`