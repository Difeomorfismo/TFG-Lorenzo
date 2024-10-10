import pygame
import pymunk
import time
import random
import pymunk.pygame_util
from math import cos,sin,pi,sqrt,dist

WIDTH=1920
HEIGHT=1080
fps=120
dt=(1/fps)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
global running
running = True

class Ball:
    def __init__(self,space,x,y,r,e):
        self.body = pymunk.Body(1,100,body_type=pymunk.Body.DYNAMIC)
        self.body.position=(int(x),int(y))
        self.shape = pymunk.Circle(self.body,10)
        self.shape.elasticity=1
        self.shape.color=(255, 0,0, 255)
        self.radius=10
        space.add(self.body,self.shape)
    def draw(self):
        pygame.draw.circle(screen, self.shape.color, self.body.position, self.radius)
class Barra:
    def __init__(self,space):        
        self.x=WIDTH/2
        self.l=100
        self.y=HEIGHT*8/9-self.l*sin(pi/4)-10
        
        self.body=pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position=(int(self.x),int(self.y))
        self.P1=(-self.l,0)
        self.P2=(self.l,0)
        self.shape = pymunk.Segment(self.body,self.P1,self.P2,radius=5)
        self.shape.color=(255, 255, 255, 255)
        self.shape.elasticity=0.8
        space.add(self.body,self.shape)
    
    
    def draw(self):
        pygame.draw.circle(screen,'white',(self.body.position),10)
        
class Obstacle:
    def __init__(self):
        r1=random.random()
        r2=random.random()
        
        self.x=(WIDTH*15/16-40)*r1 + (1-r1)*(WIDTH/16+40)
        self.y=r2*HEIGHT*5/9 + (1-r2)*(HEIGHT/9+40)
        self.r=25
        self.color=(0, 255, 255, 255)
        self.contrast=0.5
    def check_collision(self,ball):
        if dist((self.x,self.y),ball.body.position)<=self.r+ball.radius:
            return True
        else:
            return False
    def draw(self):
        pygame.draw.circle(screen,self.color,(self.x,self.y),self.r)
        pygame.draw.circle(screen,(self.color[0]*self.contrast,self.color[1]*self.contrast,self.color[2]*self.contrast),(self.x,self.y),self.r,4)
    def restart(self):
        r1=random.random()
        r2=random.random()
        self.x=(WIDTH*15/16-40)*r1 + (1-r1)*(WIDTH/16+40)
        self.y=r2*HEIGHT*5/9 + (1-r2)*(HEIGHT/9+40)

def drawX():
    global running
    pos=pygame.mouse.get_pos()
    color='grey'
    s=0
    rect=pygame.rect.Rect(WIDTH-110, 20, 60, 60)
    if rect.collidepoint(pos):
        color='red'
        s=3
        if pygame.mouse.get_pressed()[0]:
            running=False
    pygame.draw.line(screen,color,(WIDTH-100-s,70+s),(WIDTH-60+s,30-s),10+s)
    pygame.draw.line(screen,color,(WIDTH-100-s,30-s),(WIDTH-60+s,70+s),10+s)

class AddBallButton():
    def __init__(self,bolas,space):
        self.clicked=False
        self.bolas=bolas
        self.space=space
    def draw(self):
        pos=pygame.mouse.get_pos()
        color='white'
        s=0
        rect=pygame.rect.Rect(500, 20, 180, 60)
        if rect.collidepoint(pos):
            color='green'
            s=3
            if pygame.mouse.get_pressed()[0] and not self.clicked:
                self.clicked=True
                nueva_bola=Ball(self.space,WIDTH/2,HEIGHT/2,10,1)
                self.bolas.append(nueva_bola)
            elif not pygame.mouse.get_pressed()[0]:
                self.clicked=False
        score_label = pygame.font.SysFont("verdana", 30+s).render('+1 BALL',1,color)
        pygame.draw.polygon(screen, color, [(500-s,20-s),(680+s,20-s),(680+s,80+s),(500-s,80+s)],2)
        screen.blit(score_label, (520-s, 30-s))
        
class SkipButton():
    def __init__(self):
        self.clicked=False
        self.font=pygame.font.SysFont("verdana", 30)
    def draw(self):
        global score,obstacle
        pos=pygame.mouse.get_pos()
        color='white'
        s=0
        rect=pygame.rect.Rect(700, 20, 116, 60)
        if rect.collidepoint(pos):
            color='green'
            s=3
            if pygame.mouse.get_pressed()[0] and not self.clicked:
                self.clicked=True
                score-=1
                obstacle=Obstacle()
            elif not pygame.mouse.get_pressed()[0]:
                self.clicked=False
                
        score_label = self.font.render('SKIP',1,color)
        pygame.draw.polygon(screen, color, [(700-s,20-s),(816+s,20-s),(816+s,80+s),(700-s,80+s)],2)
        screen.blit(score_label, (722, 32))

class AddASpeedButton():
    def __init__(self):
        self.clicked1=False
        self.clicked2=False
        self.rect1=pygame.rect.Rect(1180, 20, 40, 27)
        self.rect2=pygame.rect.Rect(1180,53,40,27)
        self.font=pygame.font.SysFont("verdana", 30)
    def draw(self):
        global aspeed,score
        pos=pygame.mouse.get_pos()
        color='white'
        s1=0
        s2=0
        b=pygame.mouse.get_pressed()[0]
        if self.rect1.collidepoint(pos):
            color='green'
            s1=3
            if b and not self.clicked1 and aspeed<10:
                self.clicked1=True
                aspeed+=1
                score=0
            elif not b:
                self.clicked1=False
        elif self.rect2.collidepoint(pos):
            color='red'
            s2=3
            if b and not self.clicked2 and aspeed>0:
                self.clicked2=True
                aspeed-=1
                score=0
            elif not b:
                self.clicked2=False
        pygame.draw.polygon(screen, color, [(1180-s1,20-s1),(1220+s1,20-s1),(1220+s1,47+s1),(1180-s1,47+s1)],2)
        pygame.draw.line(screen,color,(1194-s1,32),(1206+s1,32),2+int(s1/3))
        pygame.draw.line(screen,color,(1200,39+s1),(1200,27-s1),2+int(s1/3))
        pygame.draw.polygon(screen, color, [(1180-s2,53-s2),(1220+s2,53-s2),(1220+s2,80+s2),(1180-s2,80+s2)],2)
        pygame.draw.line(screen,color,(1194-s2,65),(1206+s2,65),2+int(s2/3))
        
        score_label = self.font.render('Angular Speed: '+str(aspeed),1,color)
        pygame.draw.polygon(screen, color, [(850,20),(1170,20),(1170,80),(850,80)],2)
        screen.blit(score_label, (870,30))
class AddHSpeedButton():
    def __init__(self):
        self.clicked1=False
        self.clicked2=False
        self.rect1=pygame.rect.Rect(1616, 20, 40, 27)
        self.rect2=pygame.rect.Rect(1616,53,40,27)
        self.font=pygame.font.SysFont("verdana", 30)
    def draw(self):
        global hspeed,score
        pos=pygame.mouse.get_pos()
        color='white'
        s1=0
        s2=0
        b=pygame.mouse.get_pressed()[0]
        if self.rect1.collidepoint(pos):
            color='green'
            s1=3
            if b and not self.clicked1 and hspeed<15:
                self.clicked1=True
                hspeed+=1
                score=0
            elif not b:
                self.clicked1=False
        elif self.rect2.collidepoint(pos):
            color='red'
            s2=3
            if b and not self.clicked2 and hspeed>0:
                self.clicked2=True
                hspeed-=1
                score=0
            elif not b:
                self.clicked2=False
        pygame.draw.polygon(screen, color, [(1616-s1,20-s1),(1656+s1,20-s1),(1656+s1,47+s1),(1616-s1,47+s1)],2)
        pygame.draw.line(screen,color,(1630-s1,32),(1642+s1,32),2+int(s1/3))
        pygame.draw.line(screen,color,(1636,39+s1),(1636,27-s1),2+int(s1/3))
        pygame.draw.polygon(screen, color, [(1616-s2,53-s2),(1656+s2,53-s2),(1656+s2,80+s2),(1616-s2,80+s2)],2)
        pygame.draw.line(screen,color,(1630-s2,65),(1642+s2,65),2+int(s2/3))
        
        score_label = self.font.render('Horizontal Speed: '+str(hspeed),1,color)
        pygame.draw.polygon(screen, color, [(1254,20),(1606,20),(1606,80),(1254,80)],2)
        screen.blit(score_label, (1272,30))

class BallColorButton:
    def __init__(self,ball):
        self.ball=ball
        self.clicked1=False
        self.clicked2=False
        self.rect1=pygame.rect.Rect(220, 1005, 20, 40)
        self.rect2=pygame.rect.Rect(130,1005,20,40)
        self.color=self.ball.shape.color
        self.index = 0
        self.colores = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255), (0, 255, 255, 255), (255, 0, 255, 255), (255, 165, 0, 255), (0, 255, 0, 255), (255, 192, 203, 255), (238, 130, 238, 255), (64, 224, 208, 255), (255, 127, 80, 255), (255, 0, 255, 255), (50, 205, 50, 255), (65, 105, 225, 255), (255, 140, 0, 255)]

    def draw(self):
        pos=pygame.mouse.get_pos()
        color1='black'
        color2='black'
        b=pygame.mouse.get_pressed()[0]
        if self.rect1.collidepoint(pos):
            color1='white'
            if b and not self.clicked1:
                self.clicked1=True
                self.index=(self.index+1)%len(self.colores)
                self.ball.shape.color=self.colores[self.index]
                self.color=self.colores[self.index]
            elif not b:
                self.clicked1=False
        elif self.rect2.collidepoint(pos):
            color2='white'
            if b and not self.clicked2:
                self.clicked2=True
                self.index=(self.index-1)%len(self.colores)
                self.ball.shape.color=self.colores[self.index]
                self.color=self.colores[self.index]
            elif not b:
                self.clicked2=False
        #pygame.draw.polygon(screen, 'white', [(110,990),(410,990),(410,1060),(110,1060)],3)
        pygame.draw.circle(screen,self.color,(185,1025),10)
        
        pygame.draw.polygon(screen, color1, [(220,1005),(220,1045),(240,1025)])
        pygame.draw.polygon(screen, 'white', [(220,1005),(220,1045),(240,1025)],3)
        
        pygame.draw.polygon(screen, color2, [(150,1005),(150,1045),(130,1025)])
        pygame.draw.polygon(screen, 'white', [(150,1005),(150,1045),(130,1025)],3)     

class ObsColorButton:
    def __init__(self,obstacle):
        self.obs=obstacle
        self.clicked1=False
        self.clicked2=False
        self.rect1=pygame.rect.Rect(370, 1005, 20, 40)
        self.rect2=pygame.rect.Rect(280,1005,20,40)
        self.color=self.obs.color
        self.index = 4
        self.colores = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255), (0, 255, 255, 255), (255, 0, 255, 255), (255, 165, 0, 255), (0, 255, 0, 255), (255, 192, 203, 255), (238, 130, 238, 255), (64, 224, 208, 255), (255, 127, 80, 255), (255, 0, 255, 255), (50, 205, 50, 255), (65, 105, 225, 255), (255, 140, 0, 255)]

    def draw(self):
        pos=pygame.mouse.get_pos()
        color1='black'
        color2='black'
        b=pygame.mouse.get_pressed()[0]
        if self.rect1.collidepoint(pos):
            color1='white'
            if b and not self.clicked1:
                self.clicked1=True
                self.index=(self.index+1)%len(self.colores)
                self.obs.color=self.colores[self.index]
                self.color=self.colores[self.index]
            elif not b:
                self.clicked1=False
        elif self.rect2.collidepoint(pos):
            color2='white'
            if b and not self.clicked2:
                self.clicked2=True
                self.index=(self.index-1)%len(self.colores)
                self.obs.color=self.colores[self.index]
                self.color=self.colores[self.index]
            elif not b:
                self.clicked2=False
        #pygame.draw.polygon(screen, 'white', [(310,1000),(310,1050),(360,1050),(360,1000)],3)
        pygame.draw.circle(screen,self.color,(335,1025),20)
        pygame.draw.circle(screen,(self.color[0]*self.obs.contrast,self.color[1]*self.obs.contrast,self.color[2]*self.obs.contrast,255),(335,1025),20,4)
        
        pygame.draw.polygon(screen, color1, [(370,1005),(370,1045),(390,1025)])
        pygame.draw.polygon(screen, 'white', [(370,1005),(370,1045),(390,1025)],3)
        
        pygame.draw.polygon(screen, color2, [(300,1005),(300,1045),(280,1025)])
        pygame.draw.polygon(screen, 'white', [(300,1005),(300,1045),(280,1025)],3)  
        
def create_box(space):
    body1=pymunk.Body(body_type=pymunk.Body.STATIC)#techo
    shape1 = pymunk.Segment(body1,(WIDTH/16,HEIGHT/9),(WIDTH*15/16,HEIGHT/9),radius=10)
    shape1.elasticity=0.8
    shape1.color=(255, 255, 255, 255)
    space.add(body1,shape1)
    
    body2=pymunk.Body(body_type=pymunk.Body.STATIC)#derecha
    shape2 = pymunk.Segment(body2,(WIDTH*15/16,HEIGHT/9),(WIDTH*15/16,HEIGHT*8/9),radius=10)
    shape2.elasticity=0.8
    shape2.color=(255, 255, 255, 255)
    space.add(body2,shape2)
    
    
    body3=pymunk.Body(body_type=pymunk.Body.STATIC)#izquierda
    shape3 = pymunk.Segment(body3,(WIDTH*15/16,HEIGHT*8/9),(WIDTH/16,HEIGHT*8/9),radius=10)
    shape3.elasticity=0.8
    shape3.color=(255, 255, 255, 255)
    space.add(body3,shape3)
    
    
    body4=pymunk.Body(body_type=pymunk.Body.STATIC)#suelo
    shape4 = pymunk.Segment(body4,(WIDTH/16,HEIGHT*8/9),(WIDTH/16,HEIGHT/9),radius=10)
    shape4.elasticity=0.8
    shape4.friction=0
    shape4.color=(255, 255, 255, 255)
    space.add(body4,shape4)
        
def main():  
    global running, score, obstacle, aspeed, hspeed
    space = pymunk.Space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    space.gravity=(0,700)
    aspeed=5
    hspeed=5
    
    #bola=Ball(space,WIDTH/2,HEIGHT/2,10,1)
    bola=Ball(space,WIDTH/2,HEIGHT*8/9,10,1)
    Bolas=[bola]
    create_box(space)
    barra=Barra(space)
    obstacle=Obstacle()
    
    #botones:
    secret=False
    skip=SkipButton()
    addaspeed=AddASpeedButton()
    addhspeed=AddHSpeedButton()
    colorbutton=BallColorButton(bola)
    obsbutton=ObsColorButton(obstacle)
    
    black1=pygame.Rect(0,795,WIDTH/16-10,200)
    black2=pygame.Rect(1811,795,WIDTH/16-10,200)
    
    previous = time.time() * 1000
    score=0
    score_font=pygame.font.SysFont("verdana", 30)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space.remove()
        if keys[pygame.K_j] and keys[pygame.K_l]:
            barra.body.angular_velocity=0
        elif keys[pygame.K_j] and barra.body.angle>=-pi/4:
            barra.body.angular_velocity=-aspeed
        elif keys[pygame.K_l] and barra.body.angle<=pi/4:
            barra.body.angular_velocity=aspeed
        else:
            barra.body.angular_velocity=0
    
        if keys[pygame.K_a] and keys[pygame.K_d]:
            barra.body.velocity=(0,0)
        elif keys[pygame.K_a] and barra.body.position[0]>=WIDTH/16:
            barra.body.velocity=(-hspeed*100,0)
        elif keys[pygame.K_d] and barra.body.position[0]<=WIDTH*15/16:
            barra.body.velocity=(hspeed*100,0)
        else:
            barra.body.velocity=(0,0)
        for bola in Bolas:
            if obstacle.check_collision(bola)==True:
                obstacle.restart()
                score+=1
        if keys[pygame.K_b] and keys[pygame.K_o] and keys[pygame.K_l] and keys[pygame.K_a] and keys[pygame.K_SPACE]:
            secret=True
            add_ball=AddBallButton(Bolas,space)
        
        
        
        screen.fill('black')
        space.debug_draw(draw_options)
        for bola in Bolas:
            bola.draw()
        barra.draw()
        obstacle.draw()
        
        pygame.draw.rect(screen, 'black', black1)
        pygame.draw.rect(screen, 'black', black2)
        
        drawX()
        if secret:
            add_ball.draw()
        addaspeed.draw()
        addhspeed.draw()
        colorbutton.draw()
        obsbutton.draw()
        skip.draw()
        

                         
        score_label = score_font.render('SCORE: '+str(score),1,(255,255,255))
        pygame.draw.polygon(screen, 'white', [(120,20),(290+19*len(str(score)),20),(290+19*len(str(score)),80),(120,80)],2)
        screen.blit(score_label, (140, 30))
        
       
        space.step(1/120)
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