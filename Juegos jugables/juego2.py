import pygame
import time
import random
from math import sin,cos,pi

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
        
    def update(self):
        if not self.vacio:
            self.x-=self.speed
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
    def update(self,player,win):
        rectlist=[]
        for obstacle in self.list:
            if obstacle.x<-obstacle.tam:
                self.tam-=1
                obstacle.x=self.width
                obstacle.vacio=True
            rectlist.append(obstacle.rect)
            obstacle.update()
        if player.rect.collidelist(rectlist)!=-1:
            player.is_dead=True
    def draw(self,win):
        for obstacle in self.list:
            obstacle.draw(win)

def main():
    WIDTH=1600
    HEIGHT=900
    fps=120
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    running = True
    player=Player((WIDTH/3,2*HEIGHT/3))
    obstacles=ObstacleList(6,WIDTH,HEIGHT)
    pygame.init()
    previous = time.time() * 1000
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
                  
        keys = pygame.mouse.get_pressed()
        if keys[0]:
            player.jumping=True
        else:
            player.jumping=False
        if keys[2]:
            pass
            
        
        screen.fill("black")
        
        obstacles.update(player,screen)
        obstacles.spawn()
        
        player.update()
        
        obstacles.draw(screen)
        player.draw(screen)
        
        pygame.draw.line(screen,'royalblue',(0,2*HEIGHT/3),(WIDTH,2*HEIGHT/3),5)
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
    main()