import time
import pygame
import os
os.environ["SDL_VIDEO_CENTERED"] = "1"

screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Time")
clock = pygame.time.Clock()

pygame.init()

clock = pygame.time.Clock()

running = True       
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            break
        if event.type == pygame.KEYDOWN:
            # detect key 'a'
            if event.key == pygame.K_a: # key 'a'
                t = time.time()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a: # key 'a'
                t = time.time() - t; t = str(t); t = t[:5]
                print("You pressed key 'a' for",t,'seconds')


        screen.fill((255, 255, 255))
        pygame.display.update()

        clock.tick(40)