import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import os  
import cv2 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np 
import datetime

class GeneticAlgorithmGUI:
    def __init__(self, master):
        self.master = master
        master.title("Algoritmo Genético")
        master.geometry("1200x800")
        master.resizable(True, True)

        master.grid_columnconfigure(0, weight=1)  
        master.grid_columnconfigure(1, weight=1)  
        master.grid_rowconfigure(0, weight=1)     
        master.grid_rowconfigure(1, weight=0)     
        
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TLabel", font=("Arial", 12), foreground="#333333")
        style.configure("TButton", font=("Arial", 12, "bold"), foreground="#ffffff", background="#5a9", borderwidth=1, relief="raised")
        style.map("TButton", background=[("active", "#48a")])
        style.configure("TLabelframe", font=("Arial", 14, "bold"), foreground="#333333", background="#f7f7f7")
        style.configure("TLabelframe.Label", font=("Arial", 14, "bold"), foreground="#333333")
        style.configure("TEntry", font=("Arial", 12))

        input_frame = ttk.LabelFrame(master, text="Parámetros de entrada")
        input_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        output_frame = ttk.LabelFrame(master, text="Resultados")
        output_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        output_frame.rowconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)
        output_frame.columnconfigure(0, weight=1)

        button_frame = ttk.Frame(master, relief="raised", borderwidth=2)
        button_frame.grid(row=1, column=0, columnspan=2, pady=15, sticky="ew")

        self.params = {
            #"function_str": ("Función objetivo (f(x))", tk.StringVar(), "0.1 * x * math.log(1 + abs(x)) * math.cos(x) * math.cos(x)", "Función a optimizar"),
            "interval_a": ("Intervalo menor (A)", tk.StringVar(), "", "Valor mínimo del intervalo"),
            "interval_b": ("Intervalo mayor (B)", tk.StringVar(), "", "Valor máximo del intervalo"),
            "delta_x": ("Resolución (Δx)", tk.StringVar(), "", "Incremento entre puntos"),
            "pcruza": ("Probabilidad de cruza (Pcruza)", tk.StringVar(), "", "Probabilidad de cruza"),
            "p_mutation_ind": ("Probabilidad de mutación de individuos", tk.StringVar(), "", "Probabilidad de mutación de individuo"),
            "p_mutation_bit": ("Probabilidad de mutación de bits", tk.StringVar(), "", "Probabilidad de mutación de bit"),
            "max_generations": ("Número de generaciones", tk.StringVar(), "", "Iteraciones máximas"),
            "min_population_size": ("Tamaño mínimo de la población", tk.StringVar(), "", "Población mínima"),
            "max_population_size": ("Tamaño máximo de la población", tk.StringVar(), "", "Población máxima")
        }

        for i, (label_txt, var, default, tooltip) in enumerate(self.params.values()):
            label = ttk.Label(input_frame, text=label_txt + ":")
            label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
            entry = ttk.Entry(input_frame, textvariable=var, width=35)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
            var.set(str(default))

        # Añadir checkbox para minimizar o maximizar
        self.minimize_var = tk.BooleanVar(value=False)  # Si es True, se minimiza, si es False, se maximiza
        minimize_checkbox = ttk.Checkbutton(input_frame, text="Minimizar función", variable=self.minimize_var)
        minimize_checkbox.grid(row=len(self.params), column=0, columnspan=2, pady=10, sticky="w")

        self.text_results = tk.Text(output_frame, height=8, width=50, font=("Arial", 12), bg="#ffffff", fg="#333333", wrap="word", relief="solid", borderwidth=2)
        self.text_results.grid(row=0, column=0, padx=15, pady=15, sticky="ew")

        self.table_frame = ttk.Frame(output_frame)
        self.table_frame.grid(row=1, column=0, padx=15, pady=15, sticky="ew")

        # Crear un Notebook para separar las gráficas
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.grid(row=2, column=0, padx=15, pady=15, sticky="ew")

        # Crear dos frames, uno para cada gráfica
        self.graph_frame_fitness = ttk.Frame(self.notebook)
        self.graph_frame_objective = ttk.Frame(self.notebook)

        # Agregar los frames como pestañas
        self.notebook.add(self.graph_frame_fitness, text="Evolución del Fitness")
        self.notebook.add(self.graph_frame_objective, text="Función Objetivo")

        start_button = ttk.Button(button_frame, text="Iniciar Algoritmo", command=self.start_algorithm)
        start_button.grid(row=0, column=0, padx=20, pady=10, ipadx=20, sticky="ew")

        clear_button = ttk.Button(button_frame, text="Limpiar", command=self.clear_interface)
        clear_button.grid(row=0, column=1, padx=20, pady=10, ipadx=20, sticky="ew")

    def pair_selection(self, evaluated_population, pcruza):
        # Verificar si el checkbox está marcado para minimizar
        if self.minimize_var.get():  # Si se marca el checkbox, se minimiza
            evaluated_population.sort(key=lambda ind: ind[1], reverse=False)  # Ordena de menor a mayor (minimizar)
        else:  # Si no está marcado, se maximiza
            evaluated_population.sort(key=lambda ind: ind[1], reverse=True)  # Ordena de mayor a menor (maximizar)
        
        pairs = []
        for i, (individual, fitness, x) in enumerate(evaluated_population):
            p = random.random()
            if p <= pcruza:
                j = random.randint(0, i)
                pairs.append((individual, evaluated_population[j][0]))
        return pairs  

    def plot_graph(self):
        for widget in self.graph_frame_fitness.winfo_children():
            widget.destroy() 

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(self.generations, self.best_fitness_values, label='Mejor aptitud', color='green')
        ax.plot(self.generations, self.worst_fitness_values, label='Peor aptitud', color='red')
        ax.plot(
            self.generations,
            [sum(self.best_fitness_values[:i + 1]) / (i + 1) for i in range(len(self.best_fitness_values))],
            label='Aptitud promedio',
            color='blue',
            linestyle='dashed'
        )

        ax.set_xlabel("Generaciones")
        ax.set_ylabel("Aptitud")
        ax.set_title("Evolución del Fitness")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame_fitness)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()


    def show_tooltip(self, event, text):
        x, y, _, _ = event.widget.bbox("insert")
        x += event.widget.winfo_rootx() + 25
        y += event.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(event.widget)
        self.tooltip_window.wm_overrideredirect(True)
        label = tk.Label(self.tooltip_window, text=text, background="#ffffdd", relief="solid", borderwidth=1, padx=5, pady=5)
        label.pack()
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

    def calculate_n_and_bits(self):
        A = float(self.params["interval_a"][1].get())
        B = float(self.params["interval_b"][1].get())
        delta_x = float(self.params["delta_x"][1].get())

        n = (B - A) / delta_x + 1
        n = round(n)
        bits = math.ceil(math.log2(n))
        return n, bits

    def calculate_delta_x_estrella(self, bits):
        A = float(self.params["interval_a"][1].get())
        B = float(self.params["interval_b"][1].get())
        delta_x_star = (B - A) / (2 ** bits - 1)
        return delta_x_star

    def calculate_x(self, individual_index, delta_x_star):
        A = float(self.params["interval_a"][1].get())
        return A + individual_index * delta_x_star

    def validate_function(self):
        try:
        # Función fija directamente en el código, ya no depende de la entrada del usuario
            self.f_x = lambda x: 0.1 * x * math.log(1 + abs(x)) * math.cos(x) * math.cos(x)
            self.f_x(1)  # Verifica que la función se pueda evaluar con un valor
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Función inválida. Verifica la sintaxis: {e}")
            return False

    #Modelado
    def generate_initial_population(self, population_size, bits):
        population = []
        for _ in range(population_size):
            individual = "".join(random.choice(["0", "1"]) for _ in range(bits))
            population.append(individual)
        return population

    #Modelado
    def evaluate_population(self, population, bits, delta_x_star):
        #0.1 * x * math.log(1 + abs(x)) * math.cos(x) * math.cos(x)
        evaluated_population = []
        for individual in population:
            if int(individual, 2) < 2 ** bits:
                x_val = self.calculate_x(int(individual, 2), delta_x_star)
                fx = self.f_x(x_val)
                evaluated_population.append((individual, fx, x_val))
        return evaluated_population

    #Formación de parejas
    def pair_selection(self, evaluated_population, pcruza):
        evaluated_population.sort(key=lambda ind: ind[1], reverse=True)
        pairs = []
        for i, (individual, fitness, x) in enumerate(evaluated_population):
            p = random.random()
            if p <= pcruza:
                j = random.randint(0, i)
                pairs.append((individual, evaluated_population[j][0]))
        return pairs

    #Cruza
    def crossover(self, pairs, bits):
        descendants = []
        for individual1, individual2 in pairs:
            l = random.randint(0, bits - 2)
            descendant1 = individual1[:l] + individual2[l:]
            descendant2 = individual2[:l] + individual1[l:]
            descendants.append(descendant1)
            descendants.append(descendant2)
        return descendants

    #Mutación
    def mutation(self, descendants, p_mutation_ind, p_mutation_bit, bits):
        mutated_descendants = []
        for descendant in descendants:
            p = random.random()
            if p <= p_mutation_ind:
                mutated_descendant = ""
                for bit in descendant:
                    p_bit = random.random()
                    if p_bit <= p_mutation_bit:
                        mutated_descendant += "1" if bit == "0" else "0"
                    else:
                        mutated_descendant += bit
                mutated_descendants.append(mutated_descendant)
            else:
                mutated_descendants.append(descendant)
        return mutated_descendants

    #Poda
    def poda(self, current_population, descendants, bits, delta_x_star, population_size):
        combined_population = current_population + descendants
        evaluated_combined_population = self.evaluate_population(combined_population, bits, delta_x_star)
        evaluated_combined_population.sort(key=lambda ind: ind[1], reverse=True)

        unique_evaluated_population = []
        added = set()
        for individual, fitness, x in evaluated_combined_population:
            if individual not in added:
                unique_evaluated_population.append((individual, fitness, x))
                added.add(individual)

        pruned_population = [individual for individual, fitness, x in unique_evaluated_population[:population_size]]
        return pruned_population

    def plot_generation_population(self, generation, evaluated_population, bits, delta_x_star):
        A = float(self.params["interval_a"][1].get())
        B = float(self.params["interval_b"][1].get())

        fig, ax = plt.subplots(figsize=(6, 4))

        # Graficar f(x)
        x_vals = np.linspace(A, B, 200)
        y_vals = [self.f_x(x) for x in x_vals]
        ax.plot(x_vals, y_vals, label="f(x)", color="black", linewidth=1)

        # Ordenar población por fitness
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        best_ind = evaluated_population[0]
        worst_ind = evaluated_population[-1]

        pop_x = [ind[2] for ind in evaluated_population]
        pop_y = [ind[1] for ind in evaluated_population]

        ax.scatter(pop_x, pop_y, color='blue', label='Individuos', s=30)
        ax.scatter([best_ind[2]], [best_ind[1]], color='green', label='Mejor', s=80, marker='^')
        ax.scatter([worst_ind[2]], [worst_ind[1]], color='red', label='Peor', s=80, marker='v')

        ax.set_title(f"Generación {generation}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

        # Guardar figura como PNG en la carpeta de frames actual
        filename = os.path.join(self.frames_dir, f"generation_{generation}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def create_video_from_frames(self, output_filename="resultado.mp4", fps=2):
        images = [img for img in os.listdir(self.frames_dir) if img.endswith(".png")]
        # Importante: ordenar por número de generación
        images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        if not images:
            return

        first_frame_path = os.path.join(self.frames_dir, images[0])
        frame = cv2.imread(first_frame_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        for img_name in images:
            img_path = os.path.join(self.frames_dir, img_name)
            frame = cv2.imread(img_path)
            video.write(frame)

        video.release()
        cv2.destroyAllWindows()

    def plot_objective_function(self, evaluated_population_start, evaluated_population_end, best_x, best_fitness):
        A = float(self.params["interval_a"][1].get())
        B = float(self.params["interval_b"][1].get())

        # Crear figura y eje
        fig, ax = plt.subplots(figsize=(4, 3))

        # Graficar la función objetivo
        x_vals = np.linspace(A, B, 200)
        y_vals = [self.f_x(x) for x in x_vals]
        ax.plot(x_vals, y_vals, label="f(x)", color="blue")

        # Peor x inicial
        worst_start = min(evaluated_population_start, key=lambda ind: ind[1])
        worst_x_start = worst_start[2]
        worst_f_start = worst_start[1]

        # Peor x final
        worst_end = min(evaluated_population_end, key=lambda ind: ind[1])
        worst_x_end = worst_end[2]
        worst_f_end = worst_end[1]

        # Marcar puntos en la gráfica
        ax.scatter([best_x], [best_fitness], color="red", label=f"Mejor X = {best_x:.4f}", s=80, marker="o")
        ax.scatter([worst_x_start], [worst_f_start], color="black", label=f"Peor X Inicial = {worst_x_start:.4f}", s=80, marker="o")
        ax.scatter([worst_x_end], [worst_f_end], color="yellow", label=f"Peor X Final = {worst_x_end:.4f}", s=80, marker="o")

        # Personalizar la gráfica
        ax.set_title("Gráfica de la Función")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

        # Mostrar la gráfica en la interfaz
        for widget in self.graph_frame_objective.winfo_children():  # Usar graph_frame_objective
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame_objective)  # Usar graph_frame_objective
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def start_algorithm(self):
        if not self.validate_function():
            return

        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frames_dir = f"frames_{run_id}"
        os.makedirs(self.frames_dir, exist_ok=True)
        

        n, bits = self.calculate_n_and_bits()
        delta_x_star = self.calculate_delta_x_estrella(bits)

        min_population = int(self.params["min_population_size"][1].get())
        max_population = int(self.params["max_population_size"][1].get())
        max_generations = int(self.params["max_generations"][1].get())

        delta_x = float(self.params["delta_x"][1].get())

        k_population = 5
        population_size = max(min_population, min(max_population, int(k_population / delta_x) + random.randint(0, 10)))

        pcruza = float(self.params["pcruza"][1].get())
        p_mutation_ind = float(self.params["p_mutation_ind"][1].get())
        p_mutation_bit = float(self.params["p_mutation_bit"][1].get())

        current_population = self.generate_initial_population(population_size, bits)
        self.generations = []
        self.best_fitness_values = []
        self.best_individual_x_each_generation = []
        self.worst_fitness_values = []
        self.worst_individual_x_each_generation = []

        for generation in range(max_generations):
            evaluated_population = self.evaluate_population(current_population, bits, delta_x_star)
            pairs = self.pair_selection(evaluated_population, pcruza)
            descendants = self.crossover(pairs, bits)
            mutated_descendants = self.mutation(descendants, p_mutation_ind, p_mutation_bit, bits)
            current_population = self.poda(current_population, mutated_descendants, bits, delta_x_star, population_size)

            # Re-evaluar la población tras la poda
            evaluated_population_now = self.evaluate_population(current_population, bits, delta_x_star)
            evaluated_population_now.sort(key=lambda x: x[1], reverse=True)

            # Mejor y peor de la generación
            best_individual_info = evaluated_population_now[0]
            worst_individual_info = evaluated_population_now[-1]

            best_fitness = best_individual_info[1]
            best_x = best_individual_info[2]
            worst_fitness = worst_individual_info[1]
            worst_x = worst_individual_info[2]

            self.generations.append(generation)
            self.best_fitness_values.append(best_fitness)
            self.best_individual_x_each_generation.append(best_x)
            self.worst_fitness_values.append(worst_fitness)
            self.worst_individual_x_each_generation.append(worst_x)

            self.plot_generation_population(generation, evaluated_population_now, bits, delta_x_star)

        # Mejor generación
        best_generation_idx = np.argmax(self.best_fitness_values)
        best_generation = self.generations[best_generation_idx]
        best_fitness = self.best_fitness_values[best_generation_idx]
        best_x = self.best_individual_x_each_generation[best_generation_idx]

        # Evaluar al terminar las generaciones
        evaluated_population_now = self.evaluate_population(current_population, bits, delta_x_star)
        evaluated_population_now.sort(key=lambda x: x[1], reverse=True)
        best_individual_info = evaluated_population_now[0]
        self.best_individual_binary = best_individual_info[0]
        self.best_individual_fitness = best_individual_info[1]
        self.best_individual_x = best_individual_info[2]

        self.update_outputs(n, bits, delta_x_star, best_generation, best_x, best_fitness)

        self.plot_graph()

        self.plot_objective_function(
            evaluated_population_start=evaluated_population,  # Población inicial
            evaluated_population_end=evaluated_population_now,  # Población final
            best_x=self.best_individual_x,  # Mejor x
            best_fitness=self.best_individual_fitness  # Mejor f(x)
        )

        video_name = f"resultado_{run_id}.mp4"
        self.create_video_from_frames(output_filename=video_name, fps=2)
        messagebox.showinfo("Video Generado", f"Se ha creado el video '{video_name}' con las gráficas de cada generación.")


    def update_outputs(self, n, bits, delta_x_star, best_generation, best_x, best_fitness):
        self.text_results.delete("1.0", tk.END)
        self.text_results.insert(tk.END, f"Numero de puntos: {n}\n")
        self.text_results.insert(tk.END, f"Resolución inicial (Δx): {float(self.params['delta_x'][1].get()):.4f}\n")
        self.text_results.insert(tk.END, f"Resolución ajustada (Δx*): {delta_x_star:.4f}\n")
        self.text_results.insert(tk.END, f"Numero de bits: {bits}\n")
        self.text_results.insert(tk.END, f"Mejor Generación: {best_generation}\n")
        self.text_results.insert(tk.END, f"Mejor x={self.best_individual_x:.4f}\n")
        self.text_results.insert(tk.END, f"Mejor f(x)={self.best_individual_fitness:.4f}\n")
        self.text_results.insert(tk.END, f"Cadena de bits: {self.best_individual_binary} \n")
    
        self.create_table_output()

    def create_table_output(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        tree = ttk.Treeview(self.table_frame, columns=("Generación", "Mejor x", "Mejor f(x)", "Peor x", "Peor f(x)"), show="headings")
        tree.heading("Generación", text="Generación")
        tree.heading("Mejor x", text="Mejor x")
        tree.heading("Mejor f(x)", text="Mejor f(x)")
        tree.heading("Peor x", text="Peor x")
        tree.heading("Peor f(x)", text="Peor f(x)")

        for gen, best_x, best_fitness, worst_x, worst_fitness in zip(
            self.generations,
            self.best_individual_x_each_generation,
            self.best_fitness_values,
            self.worst_individual_x_each_generation,
            self.worst_fitness_values
        ):
            tree.insert("", "end", values=(gen, f"{best_x:.2f}", f"{best_fitness:.2f}", f"{worst_x:.2f}", f"{worst_fitness:.2f}"))

        tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar_y = ttk.Scrollbar(self.table_frame, orient="vertical", command=tree.yview)
        tree_scrollbar_y.grid(row=0, column=1, sticky="ns")
        tree_scrollbar_x = ttk.Scrollbar(self.table_frame, orient="horizontal", command=tree.xview)
        tree_scrollbar_x.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)

    def clear_interface(self):
        for _, var, default, _ in self.params.values():
            var.set("")
        self.text_results.delete("1.0", tk.END)

        self.generations = []
        self.best_fitness_values = []
        self.best_individual_x_each_generation = []
        self.worst_fitness_values = []
        self.worst_individual_x_each_generation = []
        self.f_x = None

        # Opcional: borrar frames previos
        for f in os.listdir(self.frames_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(self.frames_dir, f))

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()
