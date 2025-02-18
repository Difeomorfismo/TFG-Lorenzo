import pygame
import time
import random
import math

import colorsys
import torch
import torch.nn as nn
import torch.optim as optim

def get_data():
    X= []
    Y=[]
    with open('training_data.txt', 'r') as archivo:
        for linea in archivo:
            # Eliminar el salto de línea y dividir por el delimitador
            valores = linea.strip().split(';')
            # Convertir cada valor a float
            valores = [float(valor) for valor in valores]
            X.append([valores[0],valores[1]])
            Y.append([valores[2]])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Definimos la red neuronal
class Red(nn.Module):
    def __init__(self):
        super(Red, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Capa oculta con 10 neuronas
        self.fc2 = nn.Linear(10,10)  # Capa oculta con 10 neuronas
        self.fc3 = nn.Linear(10, 1)  # Capa de salida con 1 neurona

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Función de activación ReLU en la primera capa oculta
        x = torch.relu(self.fc2(x))  # Función de activación ReLU en la segunda capa oculta
        x = torch.sigmoid(self.fc3(x))  # Sin función de activación sigmoide en la capa de salida
        return x

def random_color():
    # Hue can be any value between 0 and 1
    h = random.random()
    # Saturation and Lightness are set high to ensure brightness
    s = random.uniform(0.7, 1.0)
    l = random.uniform(0.5, 0.7)
    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    # Convert RGB values from 0-1 range to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))

def hay_corte(a,b,pr,c,d,r,m):
    corte=[]
    for i in [-1,1]:
        x=a-i*math.sqrt(2)/2*pr
        y=b-m*i*math.sqrt(2)/2*pr
        k=x+m*(y-d)
        corte.append((2*c*k)+(2*r**2)-(c**2)-(k**2)>=0)
    return corte

def distancia_choque(a,b,pr,c,d,r,m,corte):
    distancias=[]
    for i in [-1,1]:
        if corte[int(i/2+1/2)]==0:
            distancias.append(math.inf)
        else:
            x=a-i*math.sqrt(2)/2*pr
            y=b-m*i*math.sqrt(2)/2*pr
            k=x+m*(y-d)
            x1=(2*(c+k)+math.sqrt(4*((2*c*k)+(2*r**2)-(c**2)-(k**2))))/4
            y1=x1-x+y
            x2=(2*(c+k)-math.sqrt(4*((2*c*k)+(2*r**2)-(c**2)-(k**2))))/4
            y2=x2-x+y
            
            if x1>=x:
                d1=(x-x1)**2+(y-y1)**2
            else:
                d1=math.inf
            if x2>=x:
                d2=(x-x2)**2+(y-y2)**2
            else:
                d2=math.inf
            distancias.append(min(d1,d2))
    return math.sqrt(min(distancias))

class Player:
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.x=width/2
        self.y=height/2
        
        self.r=10
        self.color='red'
        self.speed=4
        
        self.trail_speed=4
        self.trail_width=7
        self.trail_max_length=500
        self.vertices=[[self.x,self.y] for i in range(self.trail_max_length)]
        self.v_index=1
        
        self.click=False
        self.nerf=0
        
        self.is_dead=False
        
    def add_change(self):
        self.vertices[self.v_index]=[self.x,self.y]
        self.v_index+=1
        self.v_index%=self.trail_max_length
    
    def update(self):
        if self.click==True:
            if self.y>self.r:
                self.y-=self.speed
            else:
                self.is_dead=True
        else:
            if self.y<self.height-self.r:
                self.y+=self.speed
            else:
                self.is_dead=True

        for i in self.vertices:
            i[0]-=self.trail_speed
        self.vertices[self.v_index]=[self.x,self.y]
        
    def draw(self,win):
        if not self.is_dead:
            pygame.draw.lines(win,'white',False,self.vertices[self.v_index+1:self.trail_max_length]+self.vertices[0:self.v_index+1],self.trail_width)
            pygame.draw.circle(win, self.color, (self.x, self.y), self.r)
                
            
    def distancia(self,Obstacles,direccion):
        if direccion==-1:
            distancias=[math.sqrt(2*((self.height-self.y)**2))]
        if direccion==1:
            distancias=[math.sqrt(2*(self.y**2))]
        for obstacle in Obstacles.list:
            corte=hay_corte(self.x,self.y,self.r,obstacle.x,obstacle.y,obstacle.r,direccion)
            if True in corte:
                distancias.append(distancia_choque(self.x,self.y,self.r,obstacle.x,obstacle.y,obstacle.r,direccion,corte))
        return min(distancias)
class Obstacle:
    def __init__(self,x,y,radius,color):
        self.x=x
        self.y=y
        self.r=radius
        self.color=color
        
        self.speed=4
        self.is_none=False
    def colides_player(self,player):
        if not self.is_none:
            if math.dist((player.x,player.y),(self.x,self.y))<self.r+player.r:
                return True
            else:
                return False
    def none(self):
        self.is_none=True
        self.x=math.inf
    def update(self):
        if not self.is_none:
            self.x-=self.speed
    def draw(self,win):
        if not self.is_none:
            
            pygame.draw.circle(win,self.color,(self.x,self.y),self.r+4,10)
            pygame.draw.circle(win,(250,250,250),(self.x,self.y),self.r,3)
            

class ObstacleList:
    def __init__(self,n,distribution=random.uniform):
        self.n=n
        self.list=[Obstacle(0,0,0,(0,0,0)) for i in range(self.n)]
        for obstacle in self.list:
            obstacle.none()
        self.tam=0
        self.index=0
        self.rand=distribution
    def random_spawn(self,width,height):
        if random.random()<0.1 and self.tam<self.n:
            radius=self.rand(10,50)
            pos=self.rand(0,height)
            new_obstacle=Obstacle(width+radius,pos,radius,random_color())
            self.list[self.index]=new_obstacle
            self.index+=1
            self.index%=self.n
            self.tam+=1
    def update(self,player,win):
        for obstacle in self.list:
            if obstacle.x<-obstacle.r:
                self.tam-=1
                obstacle.none()
            if obstacle.colides_player(player):
                player.is_dead=True
            obstacle.update()
    def draw(self,win):
        for obstacle in self.list:
            obstacle.draw(win)


  
def main(model):

    WIDTH=1600
    HEIGHT=900
    fps=120
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    running = True
    player=Player(WIDTH,HEIGHT)
    
    obstacles=ObstacleList(100)
    
    toggle1=True #para cambiar la dirección
    toggle2=True #activar o desactivar hitbox
    show_hitboxes=False
    
    running = True
    pygame.init()
    previous = time.time() * 1000
    while running:
    #inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        distancia_abajo=player.distancia(obstacles,-1)
        distancia_arriba=player.distancia(obstacles,1)
        test_input = torch.tensor([distancia_abajo,distancia_arriba], dtype=torch.float32)
        output = model(test_input) 
        if output.item() >= 0.5:
            if player.click:
                player.click = True
                player.nerf+=1
            else:
                if player.nerf>=5:
                    player.click = True
                    player.nerf=0
                    player.add_change()
                else:
                    player.nerf+=1
        else:
            if not player.click:
                player.click = False
                player.nerf+=1
            else:
                if player.nerf>=5:
                    player.click =False
                    player.nerf=0
                    player.add_change()
                else:
                    player.nerf+=1
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            if toggle1:
                pass
            else:    
                player.speed=player.speed*(-1)+9
                toggle1=True
                player.add_change()
        else:
            toggle1=False
            
        if keys[pygame.K_r]:
            player.is_dead=False
            
        if keys[pygame.K_h]:
            if not toggle2:
                toggle2=True
                show_hitboxes=show_hitboxes*-1+1
        else:
            toggle2=False
        
        screen.fill("black")

    #obstacles
        obstacles.update(player,screen)
        obstacles.random_spawn(WIDTH, HEIGHT)
        
    #players
        player.update()
    
    
          

        obstacles.draw(screen)
        player.draw(screen)
        
        pygame.draw.line(screen,'red',(0,0),(WIDTH,0),5)
        pygame.draw.line(screen,'red',(0,HEIGHT-2),(WIDTH,HEIGHT-2),5)
    
        
        if show_hitboxes:
            pygame.draw.line(screen, 'yellow',(player.x,player.y), (player.x+distancia_abajo*(1/math.sqrt(2)), player.y+distancia_abajo*(1/math.sqrt(2))), 2)
            pygame.draw.line(screen, 'yellow',(player.x,player.y), (player.x+distancia_arriba*(1/math.sqrt(2)), player.y-distancia_arriba*(1/math.sqrt(2))), 2)
            pygame.draw.circle(screen, 'orange', (player.x+distancia_abajo*(1/math.sqrt(2)), player.y+distancia_abajo*(1/math.sqrt(2))),10)
            pygame.draw.circle(screen, 'orange', (player.x+distancia_arriba*(1/math.sqrt(2)), player.y-distancia_arriba*(1/math.sqrt(2))),10)

        pygame.display.update()
        
        current = time.time() * 1000
        elapsed = (current - previous)
        delay = 1000.0/fps - elapsed
        if delay<0:
            print("lag")
        delay = max(int(delay), 0)
        pygame.time.delay(delay)
        previous = time.time() * 1000

    pygame.quit()
if __name__=="__main__":
    model = Red()
    funcion_perdida = nn.MSELoss()
    optimizador = optim.Adam(model.parameters(), lr=0.01)
    entradas, y_esperado = get_data()
    etapas=1000
    for _ in range(etapas):
        optimizador.zero_grad()
        y_red = model(entradas)
        error = funcion_perdida(y_red, y_esperado)
        error.backward()
        optimizador.step()
    main(model)