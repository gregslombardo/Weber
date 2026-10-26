# Weber actualizado

Los cambios se prepararon a partir de la versión local `WEBER V2` del 26 de octubre de 2026.

- `Parabolic_Grids.ipynb`: se completó el cierre de la función truncada y su retorno. Se conservan las dos versiones de posiciones de etiquetas.
- `Wavefunctions.ipynb`: se conserva el cartel original. Las funciones de animación 2D y captura vuelven a estar definidas y usan firmas compatibles.
- `Weber_3D_Paridades_Reparametrizacion.ipynb`: se respetan los recortes del usuario; se sustituyó la sección de reparametrización por derivadas y se actualizaron las figuras y el explorador.
- `weber_core.py`: funciones numéricas, derivadas, figuras, capturas y render de animaciones completas, compartidas por los notebooks. Mantener este archivo junto a los notebooks.

Los ejes del explorador 3D abarcan de -25 a 25. En cuadráticas son coordenadas cartesianas x,y. En lineales son las variables nativas t,s: se resuelve directamente su ecuación diferencial y se dibuja sólo t,s >= 0. Los valores negativos quedan vacíos. Las cuatro combinaciones corresponden a las ramas de Frobenius 0 y 1/2, no a una paridad bajo t -> -t.

Las derivadas cuadráticas son dψ/dx y dψ/dy; las lineales son dΦ/dt y dΦ/ds. Se dejan vacíos los puntos donde no existe una derivada clásica o hay un extremo singular. No se interpola a través del corte.

Las figuras 3D y las animaciones normalizan cada panel por su máximo muestreado C. En las animaciones esa escala cambia con el fotograma y se muestra en el título de cada panel. La evolución es una variación de a, no una simulación de propagación longitudinal.

Las cuatro animaciones (dos 2D y dos 3D) son 3840 x 2160, 60 fps, 480 fotogramas y 8 segundos; a recorre -4 a 4. Los fotogramas se calculan individualmente. Los vídeos, junto con sus parámetros, están en `Weber_Animaciones/`. La resolución de vídeo no se deduce de los dpi de las figuras estáticas.

En los notebooks `RENDERIZAR_ANIMACIONES=False` permite abrirlos y ejecutarlos sin volver a renderizar los vídeos ya entregados. Cambiar a True para recalcularlos. Requiere ffmpeg para renderizar, además de NumPy, SciPy, Matplotlib, mpmath, Plotly, IPython y Jupyter.

Los valores y derivadas se contrastaron con fórmulas independientes de Kummer a 40 dígitos y las derivadas físicas también con diferencias finitas lejos del corte. Las fórmulas de referencia de Kummer están en [NIST DLMF 13.2](https://dlmf.nist.gov/13.2) y [13.3](https://dlmf.nist.gov/13.3). Los resultados de verificación numérica están en `Weber_3D_resultados/verificacion.json`.
