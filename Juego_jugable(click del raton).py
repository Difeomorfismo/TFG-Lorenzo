import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

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
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.vel = 1
        self.is_jumping = False
        self.trail = []
        self.hspeed = 2
        self.col=(255,255,255)
    def swap_speed(self):
        self.vel=self.vel*(-1)+3

    def update_trail(self):
        if self.is_jumping:
            self.y -= self.vel
        else:
            self.y += self.vel

        self.trail.append((self.x, self.y))
        if len(self.trail) > 1000:
            self.trail.pop(0)

        for i in range(len(self.trail)):
            x, y = self.trail[i]
            x -= self.hspeed
            self.trail[i] = (x, y)

    def draw(self):
        for i in range(len(self.trail) - 1):
            pygame.draw.line(win, self.col, (self.trail[i][0]-1,self.trail[i][1]), (self.trail[i+1][0],self.trail[i+1][1]), 10)
        pygame.draw.circle(win, RED, (self.x, self.y), self.radius)

    def get_shape(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

# Define the obstacle class
class Obstacle:
    def __init__(self):
        self.radius = random.randint(10, 50)
        self.x = WIDTH
        self.y = random.randint(self.radius, HEIGHT - self.radius)
        self.vel = 2
        self.mask = pygame.mask.from_surface(pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA))

    def draw(self):
        pygame.draw.circle(win, 'black', (self.x, self.y), self.radius)
        pygame.draw.circle(win, (255,0,0), (self.x, self.y), self.radius, width=5)
        #win.blit(pygame.image.load('moon.png'), (self.x,self.y))
    def update(self):
        self.x -= self.vel

    def get_shape(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def get_mask(self):
        return self.mask

    def check_collision(self, player):
        distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
        if distance < self.radius + player.radius:
            return True
        return False
class PowerUp:
    def __init__(self):
        self.radius = 10
        self.x = WIDTH
        self.y = random.randint(self.radius, HEIGHT - self.radius)
        self.vel = 2
        self.mask = pygame.mask.from_surface(pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA))

    def draw(self):
        pygame.draw.circle(win, 'yellow', (self.x, self.y), self.radius)
    def update(self):
        self.x -= self.vel

    def get_shape(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def get_mask(self):
        return self.mask

    def check_collision(self, player):
        distance = math.sqrt((self.x - player.x) ** 2 + (self.y - player.y) ** 2)
        if distance < self.radius + player.radius:
            return True
        
# Create instances of the player and obstacles
player = Player()

obstacles=[[i,0] for i in range(100)]
contador_obstaculos=0

powerups=[]

# Game loop
running = True
game_over = False

score = 0
start_time = pygame.time.get_ticks()
sanction_time=0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if not player.is_jumping:
                player.is_jumping = True

        if event.type == pygame.MOUSEBUTTONUP:
            player.is_jumping = False

    if not game_over:
        player.update_trail()
        if player.is_jumping:
            player.y -= player.vel
        else:
            player.y += player.vel

        if player.y < player.radius:
            player.y = player.radius
            sanction_time+=20
        elif player.y > HEIGHT - player.radius:
            player.y = HEIGHT - player.radius
            sanction_time+=20

        for elem in obstacles:
            obstacle=elem[1]
            index=elem[0]
            if obstacle!=0:
                obstacle.update()
                if obstacle.check_collision(player):
                    game_over = True
                if obstacle.x<obstacle.radius:
                    obstacles[index][1]=0

        # Generate new obstacles
        if random.random() < 0.1:
            obstacles[contador_obstaculos%100][1]=Obstacle()
            contador_obstaculos+=1
            
            
        # Update POWERUP positions and check for collisions
        for powerup in powerups:
            powerup.update()
            if powerup.check_collision(player):
                player.swap_speed()
                powerups.remove(powerup)

        # Remove POWERUP that touch the left border
        powerups = [powerup for powerup in powerups if powerup.x - powerup.radius > 0]

        # Generate new POWERUP
        if len(powerups) < 2 and random.random() < 0.001:
            powerups.append(PowerUp())

        # Clear the window
        win.fill(BLACK)
        #ship = pygame.image.load("fondo.jpg")
        #win.blit(ship, (0,0))

        # Draw player trail and player circle
        player.draw()

        # Draw obstacles
        for elem in obstacles:
            if elem[1]!=0:
                elem[1].draw()
        # Draw POWERUPS
        for powerup in powerups:
            powerup.draw()

        # Draw the score
        elapsed_time = pygame.time.get_ticks() - start_time -sanction_time
        score = elapsed_time // 10  # Time in milliseconds
        font = pygame.font.SysFont(None, 48)
        text = font.render(f"Score: {score}", True, WHITE)
        win.blit(text, (10, 10))

    else:
        # Display game over text
        font = pygame.font.SysFont(None, 48)
        text = font.render("Game Over! Press Enter to play again.", True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        win.blit(text, text_rect)

        # Check for restart
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            # Reset game variables
            player = Player()
            obstacles=[[i,0] for i in range(100)]
            contador_obstaculos=0
            game_over = False
            start_time = pygame.time.get_ticks()
            sanction_time=0

    # Update the display

    pygame.display.flip()

    # Control the frame rate
    clock.tick(120)

# Quit the game
pygame.quit()
