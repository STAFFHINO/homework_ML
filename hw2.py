import pygame as pygame
from math import hypot


def dbscan_naive(P, eps, m, distance):
    NOISE, C = 0, 0
    visited_points = set()
    clusters = {NOISE: []}

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        while neighbours:
            q = neighbours.pop()
            if q not in visited_points:
                visited_points.add(q)
                q_neighbours = region_query(q)
                if len(q_neighbours) > m:
                    neighbours.extend(q_neighbours)
            clusters[C].append(q)

    for p in P:
        if p in visited_points:
            continue
        visited_points.add(p)
        neighbours = region_query(p)
        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)
    return clusters


if __name__ == '__main__':
    points = []
    r = 10
    minPts, eps = 4, 3 * r
    colors = ['blue', 'green', 'purple', 'red', 'grey', 'pink', 'orange']
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    screen.fill('black')
    running = True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    point = pygame.mouse.get_pos()
                    points.append(point)
                    pygame.draw.circle(screen, 'yellow', point, r)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    labels = dbscan_naive(points, eps, minPts, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))
                    print(labels)
                    for i, group in enumerate(labels.values()):
                        for point in group:
                            pygame.draw.circle(screen, colors[i], point, r)

        pygame.display.flip()
    pygame.quit()
