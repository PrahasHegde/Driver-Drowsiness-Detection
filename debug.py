import sys
print("Current Python being used:")
print(sys.executable)

try:
    import pygame
    print("\nSUCCESS: Pygame is found!")
except ImportError:
    print("\nFAILURE: Pygame is NOT installed in the Python above.")