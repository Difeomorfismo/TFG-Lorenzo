import pygame
import random
import math

import neat
import os


# Initialize Pygame
pygame.init()

fotograma=0
grabar=-1
record=0
# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


#FONTS
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

generacion=0
# Set up the game window
WIDTH = 1920
HEIGHT = 1000
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Obstacle Game")

# Set up the clock to control the frame rate
clock = pygame.time.Clock()

# Define the player class
class Player:
    def __init__(self):
        self.radius = 10
        self.twidth=10
        self.x = WIDTH*2// 5
        self.y = HEIGHT // 2
        self.vel = 6
        self.is_jumping = False
        self.was_jumping=False
        self.trail = [(self.x,self.y)]
        self.hspeed = 6
        self.col='white'
        self.nerf=0
        self.l=[]
    def swap_speed(self):
        self.vel=self.vel*(-1)+18
    def set_speed(self,n):
        self.vel=n
        self.hspeed=2*n
    def swap_grav(self):
        self.vel*=-1
    def change_size(self):
        self.radius=-self.radius+15
        self.twidth=-self.twidth+15
    def update_trail(self):
        if len(self.trail) > 400:
            self.trail.pop(0)
        if self.is_jumping!=self.was_jumping:
            self.trail.append((self.x, self.y))
        for i in range(len(self.trail)):
            x, y = self.trail[i]
            x -= self.hspeed
            self.trail[i] = (x, y)

    def draw(self):
        for i in range(len(self.trail)-1):
            pygame.draw.line(win, self.col, (self.trail[i][0],self.trail[i][1]), (self.trail[i+1][0],self.trail[i+1][1]), self.twidth)
        pygame.draw.line(win, self.col, (self.trail[-1]), (self.x,self.y), self.twidth)
        pygame.draw.circle(win, RED, (self.x, self.y), self.radius)

    def get_shape(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Orb:
    def __init__(self, color, radius, effect, vel,width):
        self.radius = radius
        self.color=color
        self.effect=effect
        self.x = width
        self.y = random.randint(self.radius, HEIGHT - self.radius)
        self.vel = vel
    def __str__(self):
        return str(self.effect)+', '+str(self.vel)+', '+str(self.x)
    def set_speed(self,n):
        self.vel=2*n
    def draw(self):
        pygame.draw.polygon(win,self.color,((self.x-self.radius,self.y-self.radius),(self.x-self.radius,self.y+self.radius),(self.x+self.radius,self.y+self.radius),(self.x+self.radius,self.y-self.radius)))
        #pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
    def update(self):
            self.x -= self.vel

    def get_shape(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def check_collision(self, player):
        distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
        if distance < self.radius + player.radius:
            return True
    def apply_effect(self, player):
        if self.effect == "mini":
            player.swap_speed()
        elif self.effect == "reverse":
            player.swap_grav()
        elif self.effect == "size":
            player.change_size()
class Obstacle:
    def __init__(self):
        self.radius = random.randint(10, 50)
        self.x = WIDTH
        #self.y=int(random.gauss(HEIGHT/2,200))
        self.y = random.randint(self.radius, HEIGHT - self.radius)
        self.vel = 6
        self.color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    def draw(self):
        #pygame.draw.polygon(win, self.color, ((self.x,self.y-self.radius),(self.x-self.radius,self.y),(self.x,self.y+self.radius),(self.x+self.radius,self.y)))
        pygame.draw.circle(win, 'black', (self.x, self.y), self.radius)
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius, width=5)
        #win.blit(pygame.image.load('moon.png'), (self.x,self.y))
    def update(self):
        self.x -= self.vel

    def get_shape(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


    def check_collision(self, x,y,r):
        distance = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
        if distance < self.radius + r:
            return True
        return False
    def set_speed(self,n):
        self.vel=2*n
        

def main(genomes,config):
    nets=[]
    ge=[]
    players = []
    
    global generacion, fotograma, grabar, record
    generacion+=1
    
    for _,g in genomes:
        net=neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        players.append(Player())
        g.fitness=0
        ge.append(g)
    max_obs=80
    obstacles = []
    
    orbs_default=[Orb((200,0,255),10,'mini',6,WIDTH),Orb((255,255,0),10,'reverse',6,WIDTH),Orb((0,0,255),10,'size',6,WIDTH)]
    loaded_orbs=[]
    # Game loop
    running = True
    
    obstacle_random_ratio=0.1
    orb_random_ratio=0.01
    while running:
        # Check for events
        orbs_default=[Orb((200,0,255),10,'mini',6,WIDTH),Orb((0,0,255),10,'size',6,WIDTH)]  #,Orb((255,255,0),10,'reverse',6,WIDTH),]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type==pygame.MOUSEBUTTONDOWN:
                grabar*=-1
        if len(players)==0:
            running=False
            break
        for x, player in enumerate(players):
            maxfitness=0
            if player.is_jumping:
                player.was_jumping=True
            else:
                player.was_jumping=False
            
            y=player.y
            player.l=[y,1000-y,2000,2000]
            o1=0
            choque=0
            while choque==0 and o1<=player.y:
                for obstacle in obstacles:
                    if obstacle.check_collision(player.x+o1*6/player.vel,player.y-o1,player.radius):
                        choque=1
                o1+=10
            player.l[1]=o1
            choque=0
            o2=0
            while choque==0 and o2<=1000-player.y:
                for obstacle in obstacles:
                    if obstacle.check_collision(player.x+o2*6/player.vel,player.y+o2,player.radius):
                        choque=1
                o2+=10
            player.l[2]=o2
                        
                    
                    
            t=tuple(player.l)
            ge[x].fitness+=0.1
            Output=nets[players.index(player)].activate(t)
            if Output[0]>0:
                if player.is_jumping:
                    player.is_jumping = True
                    player.nerf+=1
                else:
                    if player.nerf>=2:
                        player.is_jumping = True
                        player.nerf=0
                    else:
                        player.nerf+=1
            else:
                if not player.is_jumping:
                    player.is_jumping = False
                    player.nerf+=1
                else:
                    if player.nerf>=2:
                        player.is_jumping =False
                        player.nerf=0
                    else:
                        player.nerf+=1
                        
            player.update_trail()                      
            if player.is_jumping:
                player.y -= player.vel
            else:
                player.y += player.vel
                
            
            
            
        for obstacle in obstacles:
            obstacle.update()
        for loaded_orb in loaded_orbs:
            loaded_orb.update()
        for x, player in enumerate(players):
            trash=[]
            if player.y < player.radius:
                player.y = player.radius
                ge[x].fitness-=10
                trash.append(x)
            elif player.y > HEIGHT - player.radius:
                player.y = HEIGHT - player.radius
                ge[x].fitness-=10
                trash.append(x)

            for obstacle in obstacles:
                if obstacle.check_collision(player.x,player.y,player.radius):
                    ge[x].fitness-=1
                    if x not in trash:
                        trash.append(x)
    
            for loaded_orb in loaded_orbs:
                if loaded_orb.check_collision(player):
                    loaded_orb.apply_effect(player)
                    loaded_orbs.remove(loaded_orb)
            if maxfitness<=ge[x].fitness:
                maxfitness=ge[x].fitness
            for x in trash:
                players.pop(x)
                nets.pop(x)
                ge.pop(x)
        for obstacle in obstacles:   
            if obstacle.x<obstacle.radius:
                #for x, player in enumerate(players):
                    #ge[x].fitness+=2
                    
                obstacle.radius=random.randint(10, 50)
                obstacle.y=random.randint(obstacle.radius, HEIGHT - obstacle.radius)
                obstacle.x=WIDTH
                obstacle.color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
            elif obstacle.x==WIDTH*2// 5:
                for x, player in enumerate(players):
                    ge[x].fitness+=5
        if len(obstacles) < max_obs and random.random() < obstacle_random_ratio:
            obstacles.append(Obstacle())
        
        for loaded_orb in loaded_orbs:
            if loaded_orb.x<loaded_orb.radius:
                loaded_orbs.remove(loaded_orb)
        
        if len(loaded_orbs) < 5 and random.random() < orb_random_ratio:
            new_orb=orbs_default[:][random.randint(0,1)]
            loaded_orbs.append(new_orb)
        
        if maxfitness>=record:
            record=maxfitness
        
            

        win.fill(BLACK)

        # Draw player trail and player circle
        for player in players:
            player.draw()
            #pygame.draw.line(win, 'yellow', (player.x, player.y), (player.x+player.l[1]*6/player.vel,player.y-player.l[1]),1)
            #pygame.draw.line(win, 'yellow', (player.x, player.y), (player.x+player.l[2]*6/player.vel,player.y+player.l[2]),1)
            
        # Draw obstacles
        for obstacle in obstacles:
            obstacle.draw()
        # Draw loaded_orbS
        for loaded_orb in loaded_orbs:
            loaded_orb.draw()
        #Text
        score_label = STAT_FONT.render("Vivos: " + str(len(players)),1,(255,255,255))
        win.blit(score_label, (10, 60))
        gen_label = STAT_FONT.render("Generación: " + str(generacion),1,(255,255,255))
        win.blit(gen_label, (10, 10))
        gen_label = STAT_FONT.render("Mejor: " + str(int(maxfitness)),1,(255,255,255))
        win.blit(gen_label, (10, 110))
        gen_label = STAT_FONT.render("Récord: " + str(int(record)),1,(255,255,255))
        win.blit(gen_label, (10, 160))
                    
            

        # Update the display
        pygame.display.update()
        if grabar==1:
            print('grabando')
            string=(str(fotograma))
            while len(string)<5:
                string='0'+string           
            pygame.image.save(win,'output/'+str(str(string)+'.jpg'))
            
            fotograma+=1
    
        # Control the frame rate
        clock.tick(120)
        


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats=neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner=p.run(main,100000)
    print('\nBest genome:\n{!s}'.format(winner))
if __name__=='__main__':
    # Determine path to configuration file. This path manipulation i
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)