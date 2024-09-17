GD 1.3 es la última versión que tengo en el ordenador de la red neuronal.
config-feedforward es el txt donde están todos los parámetros

Los cuadrados que se ven de vez en cuando modifican un poco el juego al colisionar con ellos. Son la clase Orb del código.
  Cuadrado Morado: Cambia la velocidad vertical del jugador al doble.
  Cuadrado Azul: Reduce la colisión del jugador a la mitad (se supone que iba a ser durante un tiempo y luego volver a la normalidad pero no lo implementé)
  Cuadrado Amarillo: Invierte los controles, en vez de pulsar para subir y soltar para bajar, es al revés.

  Normalmente la red neuronal responde bien a los cambios excepto el cuadrado amarillo, donde pierde por completo el control, ya que no recibe la información de que ha cambiado de
  estado pero se esperaba que lo detectase automáticamente.

Las variables o1 y o2 son de la forma ineficiente que uso para calcular la distancia a los obstáculos con dos rectas que salen del jugador, la cual consiste ir pixel a pixel comprobndo si toca algún obstáculo (Para cada píxel revisa TODOS los obstáculos).
En esta versión, en vez de comprobar cada píxel para comprobar la distancia, salta de 10 en 10, lo que posiblemente da información érronea de la distancia al obstáculo con un error máximo de 10 píxeles.

En las líneas 302 y 303 si se descomentan, dibuja las dos rectas de las que hablo y cómo colisiona con los obstáculos.

SOBRE LA RED NEURONAL
Uso la librería "neat", se inicializa en la línea 149 en forma de bucle. Player.l guarda la distancia en píxeles hasta colisionar en ambas direcciones y la altura en la que se encuentra el jugador.
en la linea 205 lo pongo en forma de tupla que es lo que admite la librería y en la 207 recibo la información de la función de activación después de que la librería haga los cálculos.

En ocasiones reduzco el fitness como en la línea 246 y 250, creo recordar que esto fue un intento dsesperado de castigar quitando 10 de fitness a aquellos jugadores que se chocaran contra el techo o el suelo, pero no funciona, se siguen
suicidando al comienzo contra el techo o suelo más de la mitad de los jugadores incluso 50 generaciones después. Si la muerte es por obstáculo, les quito 1 de fitness.
