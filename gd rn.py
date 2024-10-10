import pygame
import time
import random
import math
import neat
import os


import colorsys
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

def hay_corte(a,b,c,d,r,direccion):
    k=a+direccion*(b-d)
    return (2*c*k)+(2*r**2)-(c**2)-(k**2)>=0

def distancia_choque(a,b,c,d,r,direccion):
    k=a+direccion*(b-d)
    x1=(2*(c+k)+math.sqrt(4*((2*c*k)+(2*r**2)-(c**2)-(k**2))))/4
    y1=x1-a+b
    x2=(2*(c+k)-math.sqrt(4*((2*c*k)+(2*r**2)-(c**2)-(k**2))))/4
    y2=x2-a+b
    
    if x1>=a:
        d1=(a-x1)**2+(b-y1)**2
    else:
        d1=math.inf
    if x2>=a:
        d2=(a-x2)**2+(b-y2)**2
    else:
        d2=math.inf
    return math.sqrt(min(d1,d2))
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats=neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(main,1000)
    
class Player:
    def __init__(self,width,height):
        self.width=width
        self.height=height
        self.x=width/2
        self.y=height/2
        
        self.r=1
        self.color='red'
        self.speed=2
        
        self.trail_speed=2
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
            if hay_corte(self.x,self.y,obstacle.x,obstacle.y,obstacle.r,direccion):
                distancias.append(distancia_choque(self.x,self.y,obstacle.x,obstacle.y,obstacle.r,direccion))
        return min(distancias)
class Obstacle:
    def __init__(self,x,y,radius,color):
        self.x=x
        self.y=y
        self.r=radius
        self.color=color
        
        self.speed=2
        self.is_none=False
    def colides_player(self,player):
        if not self.is_none:
            if math.dist((player.x,player.y),(self.x,self.y))<=player.r+self.r:
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
        if random.random()<0.05 and self.tam<self.n:
            radius=self.rand(10,50)
            pos=self.rand(0,height)
            new_obstacle=Obstacle(width+radius,pos,radius,random_color())
            self.list[self.index]=new_obstacle
            self.index+=1
            self.index%=self.n
            self.tam+=1
    def update(self,player,x,ge,win):
        for obstacle in self.list:
            if obstacle.x<-obstacle.r:
                ge[x].fitness+=1
                self.tam-=1
                obstacle.none()
            if obstacle.colides_player(player):
                player.is_dead=True
    def draw(self,win):
        for obstacle in self.list:
            obstacle.update()
            obstacle.draw(win)


generacion=0
def main(genomes,config):
    global generacion
    generacion+=1
    nets=[]
    ge=[]
    players = []
    
    WIDTH=1600
    HEIGHT=900
    fps=120
    
    for _,g in genomes:
        net=neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        players.append(Player(WIDTH,HEIGHT))
        g.fitness=0
        ge.append(g)
    
    

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    obstacles=ObstacleList(100)
    
    running = True
    pygame.init()
    previous = time.time() * 1000
    while running:
    #inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()

        if len(players)==0:
            break
        for x, player in enumerate(players):
            ge[x].fitness+=0.1
            
            distancia_abajo=player.distancia(obstacles,-1)
            distancia_arriba=player.distancia(obstacles,1)
            player.info=[player.y,distancia_arriba,distancia_abajo]
            Output=nets[players.index(player)].activate(tuple(player.info))
            if Output[0]>0:
                if player.click:
                    player.click = True
                    player.nerf+=1
                else:
                    if player.nerf>=10:
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
                    if player.nerf>=10:
                        player.click =False
                        player.nerf=0
                        player.add_change()
                    else:
                        player.nerf+=1

            obstacles.update(player,x,ge,screen)
            if player.is_dead:
                ge[x].fitness-=1
                players.pop(x)
                print(len(players))
            player.update()
        
        screen.fill("black")

    #obstacles
        obstacles.random_spawn(WIDTH, HEIGHT)
        
          

        obstacles.draw(screen)
        for player in players:
            #pygame.draw.line(screen,'white',(player.x,player.y),(player.x+player.distancia(obstacles,1)*(1/math.sqrt(2)),player.y-player.distancia(obstacles,1)*(1/math.sqrt(2))))
            player.draw(screen)
            
        
        pygame.draw.line(screen,'red',(0,0),(WIDTH,0),5)
        pygame.draw.line(screen,'red',(0,HEIGHT-2),(WIDTH,HEIGHT-2),5)
        
        pygame.display.update()

        current = time.time() * 1000
        elapsed = (current - previous)
        delay = 1000.0/fps - elapsed
        delay = max(int(delay), 0)
        pygame.time.delay(delay)
        previous = time.time() * 1000
        

    
if __name__=="__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
    
    
