import pygame
import time
import random
from math import sin,cos,pi
import copy
import torch
import torch.nn as nn

def mutar(red, tasa_mutacion=0.1):
    with torch.no_grad():
        for param in red.parameters():
            if random.random() < tasa_mutacion:
                param.data += torch.randn_like(param) * 0.1
def crossover(modelo1, modelo2):
    hijo = Red()

    for param1, param2, param_hijo in zip(modelo1.parameters(), modelo2.parameters(), hijo.parameters()):
        if random.random() < 0.5:
            param_hijo.data = param1.data.clone()
        else:
            param_hijo.data = param2.data.clone()
    
    # Mutamos al hijo
    mutar(hijo, tasa_mutacion=0.1)
    
    return hijo
class Red(nn.Module):
    def __init__(self):
        super(Red, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Capa oculta con 10 neuronas
        self.fc2 = nn.Linear(10, 10)  # Capa oculta con 10 neuronas
        self.fc3 = nn.Linear(10, 7)  # Capa oculta con 10 neuronas
        self.fc4 = nn.Linear(7, 1)  # Capa de salida con 1 neurona

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Función de activación ReLU en la capa oculta
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sin función de activación en la capa de salida
        return x



def rotar(x,y,punto,tam,tiempo,airtime):
    tx=punto[0]-x-tam//2
    ty=punto[1]-y+tam//2
    omega=pi*tiempo/airtime
    rx=cos(omega)*tx-sin(omega)*ty
    ry=cos(omega)*ty+sin(omega)*tx
    return [rx+x+tam/2,ry+y-tam/2]
    
class Player():
    def __init__(self,spawn):
        self.x=spawn[0]
        self.y=spawn[1]
        self.tam=60
        
        self.jumping=False
        self.jump_time=0
        self.jump_v=0
        self.jump_vo=-11
        self.airtime=52
        
        self.score=0
        
        self.crouch=False
        
        self.is_dead=False
        
        self.rect=pygame.Rect(self.x,self.y-self.tam,self.tam,self.tam)
        
    def update(self):
        if not self.is_dead:
            if self.jumping and self.jump_time==0:
                self.jump_time+=1
                self.jump_v=self.jump_vo
            if self.jump_time!=0:
                self.jump_v=self.jump_vo-self.jump_vo*2/(self.airtime)*(self.jump_time)
                self.y+=self.jump_v
                
                self.jump_time+=1
                self.jump_time%=self.airtime
            self.rect=pygame.Rect(self.x,self.y-self.tam,self.tam,self.tam)

    def draw(self,win):
        if not self.is_dead:
            self.score+=1
            vertices=[(self.x,self.y),(self.x,self.y-self.tam),(self.x+self.tam,self.y-self.tam),(self.x+self.tam,self.y)]
            vertices_rotados=[rotar(self.x,self.y,i,self.tam,self.jump_time,self.airtime) for i in vertices]
            pygame.draw.polygon(win, (255,255,255), vertices_rotados)
            

class Obstacle():
    def __init__(self,x,y,vacio=False):
        self.vacio=vacio
        self.x=x
        self.y=y
        self.speed=5
        self.tam=60
        self.rect=pygame.Rect(self.x+20,self.y-30,20,30)
        self.score=False
    def update(self,width,players):
        if not self.vacio:
            if self.x>width/3:
                self.score=False
            self.x-=self.speed
            if self.x<=width/3 and self.score==False:
                self.score=True
                for player in players:
                    player.score+=1
            self.rect=pygame.Rect(self.x+20,self.y-30,20,30)
    def draw(self,win):
        if not self.vacio:
            vertices=[(self.x,self.y),(self.x+self.tam,self.y),(self.x+self.tam/2,self.y-self.tam)]
            pygame.draw.polygon(win, 'red', vertices)
            #pygame.draw.rect(win,'white',self.rect)
        
        
class ObstacleList():
    def __init__(self,n,WIDTH,HEIGHT):
        self.width=WIDTH
        self.height=HEIGHT
        
        self.n=n
        self.tam=0
        self.index=0
        self.list=[Obstacle(WIDTH,HEIGHT*2/3,vacio=True) for i in range(n)]
        
        self.cooldown=0
    def spawn(self):
        if random.random()<0.02 and self.tam<self.n and self.cooldown%12==0:
            self.list[self.index].vacio=False
            self.index+=1
            self.index%=self.n
            self.tam+=1
            self.cooldown+=1
        elif self.cooldown>0:
            self.cooldown+=1
            self.cooldown%=48
    def update(self,players): 
        for obstacle in self.list:
            if obstacle.x<-obstacle.tam:
                self.tam-=1
                obstacle.x=self.width
                obstacle.vacio=True
            obstacle.update(self.width,players)
    def draw(self,win):
        for obstacle in self.list:
            obstacle.draw(win)
    def colision(self,player):
        rectlist=[obstacle.rect for obstacle in self.list]
        if player.rect.collidelist(rectlist)!=-1:
            player.is_dead=True

def main(models,N,elite_n):
    WIDTH=1600
    HEIGHT=900
    fps=120
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    running = True
    players=[]
    gen=0
    elite=[]
    for i in range(N):
        players.append(Player((WIDTH/3,2*HEIGHT/3)))
    obstacles=ObstacleList(6,WIDTH,HEIGHT)
    
    pygame.init()
    previous = time.time() * 1000
    
    model=Red()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_s] and elite!=[]:
            for i in range(elite_n):
                torch.save(elite[i].state_dict(), "modelo_pesos"+str(i)+".pth")
                pass
        for i in range(N):
            player=players[i]
            model=models[i]
            distancias=sorted([(obstacle.x-player.x) for obstacle in obstacles.list if obstacle.x-player.x>=0])[:3]
            if len(distancias)==3:
                datos=[distancias[0]-10,distancias[1]-distancias[0],distancias[2]-distancias[1]]
            elif len(distancias)==2:
                datos=[distancias[0]-10,distancias[1]-distancias[0],0]
            else:
                datos=[WIDTH,0,0]
            datos+=[(HEIGHT*2/3-player.y)]
            #print(datos)
            test_input = torch.tensor(datos, dtype=torch.float32)
            output=model(test_input)
            if output.item()>=0.5:
                player.score-=0.1
                player.jumping=True
            else:
                player.jumping=False
            
        
        screen.fill("black")

        for player in players:
            obstacles.colision(player)
            player.update()
        obstacles.update(players)
        obstacles.spawn()
        obstacles.draw(screen)
        scores=[]
        end=1
        for player in players:
            end*=player.is_dead
            scores.append(player.score)
            player.draw(screen)
        #print(sum([1 for i in players if not i.is_dead]))
        pygame.draw.line(screen,'royalblue',(0,2*HEIGHT/3),(WIDTH,2*HEIGHT/3),5)
        pygame.display.update()
        
        
        current = time.time() * 1000
        elapsed = (current - previous)
        delay = 1000.0/fps - elapsed
        if delay<0:
            #print("lag")
            pass
        delay = max(int(delay), 0)
        pygame.time.delay(delay)
        previous = time.time() * 1000
        if end==1:
            gen+=1
            elite=[models[i] for i in [x[0] for x in sorted(list(enumerate(scores)), key=lambda x: x[1],reverse=True)[0:elite_n]]]
            players=[]
            for i in range(N):
                players.append(Player((WIDTH/3,2*HEIGHT/3)))
            obstacles=ObstacleList(6,WIDTH,HEIGHT)
            hijos = []
            indice=0
            while len(hijos) + elite_n < N:
                padre1, padre2 = elite[indice],elite[(indice+1)%elite_n]
                hijo = crossover(padre1, padre2)
                hijos.append(hijo)
                indice+=1
                indice%=elite_n
            models=elite+hijos
    
    pygame.quit()
if __name__=="__main__":
    N=20
    elite_n=3 #divide a N-nuevos
    models=[]
    cargar=True
    i=0
    while i < N:
        if cargar:
            modelo = Red()
            modelo.load_state_dict(torch.load("modelo_pesos"+str(i%20)+".pth",weights_only=True))
            models.append(modelo)
        else:
            models.append(Red())
        i+=1
    main(models,N,elite_n)