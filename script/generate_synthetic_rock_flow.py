"""
generate_synthetic_rock_flow.py — Generate a synthetic video of flowing water with large rocks
========================================================================================
Synthesizes a 512x512 video containing:
1. Moving tiny tracer particles that follow a wavy flow field (allowing PIV to predict flow).
2. Large, irregular, textured rock particles flowing through the channel (for PTV tracking).
3. Realistic camera noise and blur.

Output:
    data/PTV_dataset/synthetic_rock_flow.mp4
"""

import os
import numpy as np
import cv2

# Define output path
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(WORKSPACE, 'data', 'PTV_dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'synthetic_rock_flow.mp4')

# Constants
WIDTH, HEIGHT = 512, 512
FPS = 20
NUM_FRAMES = 120
NUM_TRACERS = 1000

# Wavy flow field velocity function
def get_velocity(x, y, t):
    # Flow mostly left-to-right (u) with sinusoidal waving (v)
    u = 6.0 + 1.5 * np.sin(y / 35.0 + t / 6.0)
    v = 2.0 * np.cos(x / 50.0 + t / 10.0)
    return u, v

# Rock Class for irregular moving objects with translation and rotation
class Rock:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.base_radius = np.random.uniform(15.0, 25.0)
        self.color = int(np.random.randint(180, 240))
        
        # Rotational state
        self.theta = np.random.uniform(0, 2 * np.pi) # Initial rotation angle
        # Slower, realistic angular velocity in radians per frame (-0.08 to +0.08)
        self.omega = np.random.uniform(-0.08, 0.08) 
        
        # Pre-generate vertices shape relative to center
        num_pts = np.random.randint(7, 12)
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        radii = self.base_radius + np.random.uniform(-self.base_radius * 0.25, self.base_radius * 0.25, num_pts)
        
        self.rel_vertices = []
        for a, r in zip(angles, radii):
            vx = r * np.cos(a)
            vy = r * np.sin(a)
            self.rel_vertices.append([vx, vy])
        self.rel_vertices = np.array(self.rel_vertices, dtype=np.float32)
        
        # Pre-generate some random inner crack coordinates for texture
        self.num_cracks = np.random.randint(2, 5)
        self.crack_starts = np.random.uniform(-self.base_radius * 0.5, self.base_radius * 0.5, (self.num_cracks, 2))
        self.crack_ends = self.crack_starts + np.random.uniform(-self.base_radius * 0.4, self.base_radius * 0.4, (self.num_cracks, 2))

    def move(self, t):
        # Translate based on local flow velocity
        u, v = get_velocity(self.x, self.y, t)
        self.x += u
        self.y += v
        # Rotate
        self.theta += self.omega

    def draw(self, img):
        # Rotation matrix
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Rotate vertices
        rotated_rel = np.dot(self.rel_vertices, rot_matrix.T)
        vertices = (rotated_rel + np.array([self.x, self.y])).astype(np.int32)
        
        # Draw the rock shape (filled irregular polygon)
        cv2.fillPoly(img, [vertices], self.color)
        
        # Draw rock borders with a slightly darker shade
        cv2.polylines(img, [vertices], True, max(0, self.color - 60), 2, lineType=cv2.LINE_AA)
        
        # Rotate and draw crack lines for realistic rock texture
        rotated_starts = np.dot(self.crack_starts, rot_matrix.T)
        rotated_ends = np.dot(self.crack_ends, rot_matrix.T)
        for start, end in zip(rotated_starts, rotated_ends):
            p1 = (int(self.x + start[0]), int(self.y + start[1]))
            p2 = (int(self.x + end[0]), int(self.y + end[1]))
            cv2.line(img, p1, p2, max(0, self.color - 80), 1, lineType=cv2.LINE_AA)

def main():
    print(f"Generating synthetic rock flow video...")
    print(f"Saving to: {OUTPUT_PATH}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    # Initialize tracer particles (x, y)
    tracers = np.random.uniform(0, WIDTH, (NUM_TRACERS, 2))
    # Give tracers unique intensities to resemble natural water impurities
    tracer_colors = np.random.randint(80, 160, NUM_TRACERS)

    # Initialize rocks list
    rocks = []
    # Pre-populate some rocks across the channel
    for x in np.arange(50, WIDTH, 120):
        rocks.append(Rock(x, np.random.uniform(50, HEIGHT - 50)))

    # Frame generation loop
    for t in range(NUM_FRAMES):
        # 1. Start with a dark gray riverbed background
        frame = np.full((HEIGHT, WIDTH), 25, dtype=np.uint8)

        # 2. Update and draw tiny tracer particles
        for i in range(NUM_TRACERS):
            tx, ty = tracers[i]
            u, v = get_velocity(tx, ty, t)
            tx += u
            ty += v
            
            # Warp around boundaries if they leave the screen
            if tx >= WIDTH:
                tx = 0.0
                ty = np.random.uniform(0, HEIGHT)
            if ty < 0 or ty >= HEIGHT:
                ty = np.random.uniform(0, HEIGHT)
                tx = 0.0
                
            tracers[i] = [tx, ty]
            # Draw tiny tracer particle (1x1 or 2x2 px)
            x_int, y_int = int(tx), int(ty)
            if 0 <= x_int < WIDTH and 0 <= y_int < HEIGHT:
                frame[y_int, x_int] = tracer_colors[i]
                if x_int + 1 < WIDTH:
                    frame[y_int, x_int + 1] = tracer_colors[i] // 2

        # 3. Update and draw large rocks
        # Filter out rocks that have moved off-screen
        rocks = [r for r in rocks if r.x < WIDTH + 40]
        
        # Spawn new rocks if counts drop
        if len(rocks) < 5 and np.random.rand() < 0.15:
            rocks.append(Rock(-40, np.random.uniform(50, HEIGHT - 50)))

        for r in rocks:
            r.move(t)
            r.draw(frame)

        # 4. Apply camera optics (blur + noise)
        # Small Gaussian blur to blend particles naturally
        frame_blurred = cv2.GaussianBlur(frame, (3, 3), 0.5)
        
        # Add random camera sensor noise
        noise = np.random.normal(0, 4.0, frame.shape).astype(np.float32)
        frame_noisy = np.clip(frame_blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Convert to BGR format for video output
        frame_bgr = cv2.cvtColor(frame_noisy, cv2.COLOR_GRAY2BGR)

        # Write to video
        writer.write(frame_bgr)

        if (t + 1) % 20 == 0:
            print(f"Generated frame {t+1}/{NUM_FRAMES}")

    writer.release()
    print(f"\nSynthetic video generation complete! Saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
