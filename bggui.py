import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from io import BytesIO
import os
from coordinates import getCoordinates
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBackgroundInitiate


# Load vehicle image once
with open("assets/cars/images2_original.jpg", "rb") as f:
    detected_vehicle = BytesIO(f.read())

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Generator")
        self.root.geometry("1400x800")

        # Initialize default angle_id and coordinates
        self.angle_id = tk.StringVar(value="2")
        self.normalCoordinates = getCoordinates(self.angle_id.get(), detected_vehicle)

        self.create_widgets()

    def create_widgets(self):
        # Create main frames
        self.left_frame = ttk.Frame(self.root, width=350)
        self.left_frame.pack(side='left', fill='y')

        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side='right', fill='both', expand=True)

        self.entries = {}
        row = 0

        # Angle ID Input
        ttk.Label(self.left_frame, text="Angle ID").grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(self.left_frame, textvariable=self.angle_id)
        entry.grid(row=row, column=1)
        entry.bind("<FocusOut>", self.update_coordinates)
        row += 1

        # Input fields for normalCoordinates
        for key in self.normalCoordinates:
            ttk.Label(self.left_frame, text=key).grid(row=row, column=0, sticky="w")
            val_str = tk.StringVar(value=str(self.normalCoordinates[key]))
            entry = ttk.Entry(self.left_frame, textvariable=val_str, width=25)
            entry.grid(row=row, column=1)
            self.entries[key] = val_str
            row += 1

        # Buttons
        ttk.Button(self.left_frame, text="Generate", command=self.generate_background).grid(row=row, column=0, pady=10)
        ttk.Button(self.left_frame, text="Save", command=self.save_values).grid(row=row, column=1, pady=10)

        # Image display area
        self.image_label = ttk.Label(self.right_frame)
        self.image_label.pack(expand=True)

    def update_coordinates(self, event=None):
        try:
            new_coords = getCoordinates(self.angle_id.get(), detected_vehicle)
            for key in new_coords:
                self.entries[key].set(str(new_coords[key]))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update coordinates: {e}")

    def generate_background(self):
        try:
            coords = {key: eval(self.entries[key].get()) for key in self.entries}
            angle = "normal" if self.angle_id.get() in ["1", "2", "3", "5", "6", "7", "9", "12", "13", "16"] else "reverse"

            # Prepare required inputs for addBackgroundInitiate (mocked here)
            wallImages_bytes = {
                k: BytesIO(open(f"assets/walls/7/{name}.jpg", "rb").read())
                for k, name in {
                    "ceiling_wall_img": "ceiling",
                    "left_wall_img": "lw",
                    "right_wall_img": "rw",
                    "floor_wall_img": "floor"
                }.items()
            }

            logo_path = "assets/logo/Artboard 47.png"
            with open(logo_path, "rb") as f:
                logo_bytesio = BytesIO(f.read())

            background = addBackgroundInitiate(
                wallImages_bytes=wallImages_bytes,
                logo_bytes=logo_bytesio,
                floor_coordinates=[
                    coords["floor_left_top"],
                    coords["floor_left_bottom"],
                    coords["floor_right_bottom"],
                    coords["canvas_middle_ref"]
                ],
                left_wall_coordinates=[
                    coords["rwall_top_left"],
                    coords["lwall_left_top"],
                    coords["lwall_left_bottom"],
                    coords["canvas_middle_ref"]
                ],
                right_wall_coordinates=[
                    coords["rwall_top_left"],
                    coords["canvas_middle_ref"],
                    coords["rwall_right_bottom"],
                    coords["rwall_top_right"]
                ],
                ceiling_coordinates=[
                    coords["lwall_left_top"],
                    coords["rwall_top_left"],
                    coords["rwall_top_right"],
                    coords["ceiling_top"]
                ],
                angle=angle,
                logo_position="auto"
            )

            img = Image.open(background)
            img = img.resize((1000, 700))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))

    def save_values(self):
        try:
            with open("./currentValues.txt", "w") as f:
                f.write(f"angle_id: {self.angle_id.get()}\n")
                for key, var in self.entries.items():
                    f.write(f"{key}: {var.get()}\n")
            messagebox.showinfo("Saved", "Values saved to ./currentValues.txt")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
