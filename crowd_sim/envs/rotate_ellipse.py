import pygame
import random

def main():
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    screen_rect = screen.get_rect()
    clock = pygame.time.Clock()
    surface = pygame.Surface((100, 200))
    surface.set_colorkey((2, 3, 4))
    surface.fill((2, 3, 4))
    rect = surface.get_rect(center=(100, 100))
    pygame.draw.ellipse(surface, pygame.Color('white'), (0, 0, 100, 200))
    angle = 0    
    dt = 0
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                return

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]: rect.move_ip(0, -5)
        if pressed[pygame.K_DOWN]: rect.move_ip(0, 5)
        if pressed[pygame.K_LEFT]: rect.move_ip(-5, 0)
        if pressed[pygame.K_RIGHT]: rect.move_ip(5, 0)
        if pressed[pygame.K_a]: angle += 1
        if pressed[pygame.K_d]: angle -= 1
  
        rotated = pygame.transform.rotate(surface, angle)
        rect = rotated.get_rect(center=rect.center)

        rect.clamp_ip(screen_rect)

        screen.fill(pygame.Color('dodgerblue'))
        screen.blit(rotated, rect.topleft)
        pygame.display.update()
        dt = clock.tick(60)

if __name__ == '__main__':
    main()
    