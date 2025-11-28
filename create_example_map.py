from PIL import Image, ImageDraw

# Create a simple example map
width, height = 20, 20
img = Image.new('RGB', (width, height), color=(255, 255, 255))  # White background
pixels = img.load()

# Draw walls (black) around border
for i in range(width):
    pixels[i, 0] = (0, 0, 0)
    pixels[i, height-1] = (0, 0, 0)
for i in range(height):
    pixels[0, i] = (0, 0, 0)
    pixels[width-1, i] = (0, 0, 0)

# Add some internal walls
for i in range(5, 15):
    pixels[i, 5] = (0, 0, 0)
    pixels[i, 14] = (0, 0, 0)

# Place Pacman (yellow)
pixels[10, 10] = (255, 255, 0)

# Place ghosts
pixels[3, 3] = (255, 0, 0)      # Red
pixels[3, 16] = (255, 192, 203)  # Pink
pixels[16, 3] = (0, 255, 255)    # Cyan
pixels[16, 16] = (255, 165, 0)   # Orange

# Place power pellets (blue) in corners
pixels[2, 2] = (0, 0, 255)
pixels[2, 17] = (0, 0, 255)
pixels[17, 2] = (0, 0, 255)
pixels[17, 17] = (0, 0, 255)

# Save
img.save('example_map.png')
print("Created example_map.png")
print("\nColor guide:")
print("- Black (0,0,0): Wall")
print("- White (255,255,255): Empty space (pellet)")
print("- Yellow (255,255,0): Pacman start")
print("- Red (255,0,0): Ghost 1 start")
print("- Pink (255,192,203): Ghost 2 start")
print("- Cyan (0,255,255): Ghost 3 start")
print("- Orange (255,165,0): Ghost 4 start")
print("- Blue (0,0,255): Power pellet")
