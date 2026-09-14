
from manim import *
import os

class CoulombWeber4K(Scene):
    def construct(self):
        image_folder = "frames_4k"
        if not os.path.exists(image_folder):
            self.add(Text("Carpeta frames_4k no encontrada").scale(0.5))
            return

        files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
        images = [ImageMobject(os.path.join(image_folder, f)).scale(2.5) for f in files]

        self.add(images[0])
        for i in range(len(images) - 1):
            self.add(images[i+1])
            self.remove(images[i])
            self.wait(1/60)
