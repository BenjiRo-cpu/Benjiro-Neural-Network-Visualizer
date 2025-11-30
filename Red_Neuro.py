import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import time
import threading
import random

class NeuralNetworkVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal Benjiro v2.3")
        self.root.geometry("1000x700")
        self.root.configure(bg='black')
        
        # Configuración principal
        self.setup_ui()
        self.setup_visualization()
        
        # Base de conocimiento para respuestas
        self.setup_knowledge_base()
        
        # Variable para controlar la animación de "pensamiento"
        self.thinking = False
    
    def setup_knowledge_base(self):
        # Base de conocimiento para preguntas y respuestas
        self.qa_database = {
            "red neuronal": [
                "Una red neuronal es un modelo computacional inspirado en el funcionamiento del cerebro humano. Está compuesta por nodos interconectados (neuronas artificiales) organizados en capas, que pueden aprender patrones complejos a partir de datos. Las redes neuronales son fundamentales en el aprendizaje profundo y han revolucionado campos como la visión por computadora, el procesamiento del lenguaje natural y la robótica.",
                "Las redes neuronales son sistemas de procesamiento de información que imitan la estructura y funcionamiento de las neuronas biológicas. Consisten en capas de nodos interconectados que procesan señales mediante funciones de activación. Su capacidad para aprender de los datos las hace extremadamente útiles para resolver problemas complejos como reconocimiento de patrones, clasificación y predicción."
            ],
            "inteligencia artificial": [
                "La Inteligencia Artificial (IA) es la disciplina que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Esto incluye razonamiento, aprendizaje, planificación, percepción y comprensión del lenguaje natural. La IA moderna utiliza principalmente algoritmos de aprendizaje automático y redes neuronales para desarrollar capacidades como reconocimiento de imágenes, traducción de idiomas y toma de decisiones.",
                "La Inteligencia Artificial comprende el desarrollo de algoritmos y sistemas que pueden simular aspectos de la inteligencia humana. Abarca desde sistemas basados en reglas hasta complejos modelos de aprendizaje profundo. Las aplicaciones de IA están transformando industrias como la medicina, finanzas, transporte y entretenimiento, permitiendo automatizar tareas complejas y descubrir patrones en grandes volúmenes de datos."
            ],
            "aprendizaje profundo": [
                "El aprendizaje profundo (Deep Learning) es una rama avanzada del aprendizaje automático que utiliza redes neuronales con múltiples capas ocultas. Esta técnica permite modelar abstracciones de alto nivel en los datos mediante arquitecturas compuestas por transformaciones no lineales. Ha logrado avances revolucionarios en reconocimiento de imágenes, procesamiento del lenguaje natural, generación de contenido y muchos otros campos.",
                "El Deep Learning representa un enfoque del aprendizaje automático basado en representaciones en capas. Las redes neuronales profundas pueden tener desde algunas hasta cientos de capas ocultas, cada una extrayendo características más abstractas de los datos. Arquitecturas como las CNN (redes convolucionales) para imágenes, las RNN y Transformers para secuencias, han permitido logros sin precedentes en IA, como los sistemas de conducción autónoma o los modelos de lenguaje avanzados."
            ],
            "neurona artificial": [
                "Una neurona artificial es la unidad básica de procesamiento en una red neuronal. Inspirada en las neuronas biológicas, recibe múltiples entradas, les asigna pesos, los combina, aplica una función de activación y produce una salida. Esta estructura simple pero poderosa permite a las redes neuronales aprender y aproximar virtualmente cualquier función matemática cuando se combinan en grandes cantidades.",
                "Las neuronas artificiales son nodos computacionales que procesan información de manera similar a las neuronas biológicas. Cada neurona recibe señales de entrada, las pondera según la importancia de cada conexión, suma estos valores y aplica una función de activación para determinar si la neurona se 'activa' y transmite una señal. El proceso de aprendizaje consiste en ajustar estos pesos para minimizar el error en las predicciones."
            ],
            "función de activación": [
                "Una función de activación determina la salida de una neurona artificial basándose en sus entradas ponderadas. Las funciones no lineales como ReLU, sigmoid y tanh son esenciales para que las redes neuronales puedan aprender patrones complejos. Sin estas funciones, una red neuronal se reduciría a una simple regresión lineal independientemente de su profundidad.",
                "Las funciones de activación introducen no-linealidad en las redes neuronales, permitiéndoles aprender relaciones complejas en los datos. Algunas funciones populares incluyen: Sigmoid (comprime valores entre 0 y 1), Tanh (comprime valores entre -1 y 1), ReLU (devuelve 0 para entradas negativas y la entrada misma para positivas) y Softmax (convierte valores en probabilidades para clasificación multiclase)."
            ],
            "entrenamiento": [
                "El entrenamiento de una red neuronal es el proceso mediante el cual el modelo aprende a realizar una tarea ajustando sus parámetros (pesos y sesgos). Utiliza conjuntos de datos etiquetados, una función de pérdida para medir el error, y algoritmos de optimización como el descenso de gradiente para minimizar este error. Durante este proceso, la red ajusta gradualmente sus parámetros para mejorar su rendimiento en la tarea asignada.",
                "Entrenar una red neuronal implica presentarle ejemplos y ajustar sus parámetros para minimizar la diferencia entre sus predicciones y los resultados esperados. Este proceso iterativo involucra: propagación hacia adelante (calcular predicciones), calcular el error con una función de pérdida, propagación hacia atrás (calcular gradientes) y actualizar los parámetros. Con suficientes datos y tiempo, la red puede aprender a generalizar a ejemplos nunca vistos."
            ],
            "backpropagation": [
                "La retropropagación (backpropagation) es el algoritmo fundamental para entrenar redes neuronales. Funciona calculando el gradiente de la función de pérdida con respecto a cada peso mediante la regla de la cadena del cálculo. Estos gradientes indican cómo ajustar los pesos para minimizar el error. El proceso se realiza de atrás hacia adelante, desde la capa de salida hasta la de entrada, actualizando todos los pesos en cada iteración.",
                "Backpropagation es el método que permite a las redes neuronales aprender, distribuyendo el error desde la salida hacia las capas anteriores. Primero, se calcula el error en la capa final comparando la salida con el valor esperado. Luego, este error se propaga hacia atrás, determinando la contribución de cada neurona al error total. Finalmente, los pesos se ajustan proporcionalmente a su contribución al error, permitiendo que la red mejore con cada iteración de entrenamiento."
            ],
            "cnn": [
                "Las Redes Neuronales Convolucionales (CNN) son arquitecturas especializadas en procesar datos con estructura de cuadrícula, como imágenes. Utilizan capas convolucionales que aplican filtros sobre los datos de entrada para detectar características locales. Esta estructura permite a las CNN identificar patrones independientemente de su posición en la imagen, haciéndolas extremadamente eficaces para tareas como reconocimiento de objetos, detección facial y segmentación de imágenes.",
                "Las CNN revolucionaron el campo de la visión por computadora mediante su estructura inspirada en el córtex visual humano. Sus componentes principales incluyen: capas convolucionales (detectan características como bordes y texturas), funciones de activación (añaden no-linealidad), capas de pooling (reducen dimensionalidad preservando información relevante) y capas totalmente conectadas (realizan la clasificación final). Esta arquitectura les permite aprender jerarquías de características, desde simples bordes hasta complejas formas y objetos."
            ],
            "rnn": [
                "Las Redes Neuronales Recurrentes (RNN) son diseñadas para procesar datos secuenciales como texto, audio o series temporales. A diferencia de las redes neuronales tradicionales, las RNN tienen conexiones que forman ciclos, permitiéndoles mantener una 'memoria' de entradas previas. Esta característica las hace ideales para tareas como traducción automática, reconocimiento de voz y generación de texto, donde el contexto temporal es crucial.",
                "Las RNN se distinguen por su capacidad para procesar secuencias de longitud variable manteniendo un estado interno que funciona como memoria. Sin embargo, enfrentan el problema de desvanecimiento del gradiente en secuencias largas. Para solucionar esto, se desarrollaron variantes como LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit), que utilizan mecanismos de compuertas para controlar el flujo de información y mantener dependencias a largo plazo en los datos."
            ],
            "lstm": [
                "Las redes LSTM (Long Short-Term Memory) son un tipo especializado de RNN diseñadas para aprender dependencias a largo plazo. Utilizan un sistema de compuertas (input, forget y output) que controlan el flujo de información, permitiendo a la red retener información relevante durante muchos pasos temporales y olvidar lo irrelevante. Las LSTM han sido cruciales para avances en traducción automática, generación de texto y reconocimiento de voz.",
                "Las LSTM resuelven el problema de la desaparición del gradiente en RNN tradicionales mediante una sofisticada arquitectura de celdas de memoria. Cada celda LSTM contiene mecanismos para decidir qué información nueva incorporar, qué información olvidar y qué parte del estado actual usar para calcular la salida. Esta estructura permite a las LSTM capturar dependencias temporales complejas en datos secuenciales, siendo particularmente efectivas en tareas que requieren memoria a largo plazo."
            ],
            "transformers": [
                "Los Transformers son arquitecturas de redes neuronales basadas en mecanismos de atención que han revolucionado el procesamiento del lenguaje natural. A diferencia de las RNN, procesan toda la secuencia simultáneamente, utilizando la atención para ponderar la importancia de diferentes palabras en el contexto. Modelos como BERT, GPT y T5 son transformers que han logrado resultados excepcionales en tareas como traducción, respuesta a preguntas y generación de texto.",
                "La arquitectura Transformer, introducida en 2017, transformó el PLN mediante su mecanismo de auto-atención que permite modelar relaciones entre todas las palabras de una secuencia en paralelo. Esto soluciona los problemas de las RNN con secuencias largas y permite un entrenamiento más eficiente. Su estructura incluye codificadores y decodificadores con múltiples capas de atención y redes feed-forward. Los Transformers son la base de los modelos de lenguaje más avanzados actualmente, impulsando aplicaciones como asistentes virtuales y sistemas de traducción."
            ],
            "gan": [
                "Las Redes Generativas Adversarias (GAN) consisten en dos redes neuronales que compiten entre sí: un generador que crea datos sintéticos y un discriminador que evalúa su autenticidad. Durante el entrenamiento, el generador mejora progresivamente para producir ejemplos más realistas, mientras el discriminador se vuelve más hábil en detectar falsificaciones. Este enfoque ha permitido generar imágenes, música y texto con un nivel de realismo sin precedentes.",
                "Las GAN representan un enfoque revolucionario para el aprendizaje no supervisado donde dos redes compiten en un juego de suma cero. El generador intenta maximizar la probabilidad de que el discriminador cometa un error, mientras el discriminador intenta distinguir correctamente entre ejemplos reales y generados. Este proceso adversario permite a las GAN capturar distribuciones de datos complejas, resultando en aplicaciones como super-resolución de imágenes, traducción imagen-a-imagen, síntesis de rostros realistas y creación de arte generativo."
            ],
        }
        
        # Respuestas para preguntas generales o no reconocidas
        self.general_responses = [
            "Esta demostración simula el funcionamiento conceptual de una red neuronal. Las redes neuronales reales procesan datos a través de múltiples capas de neuronas interconectadas, ajustando pesos mediante algoritmos de aprendizaje.",
            "En la visualización puedes observar cómo la información fluye a través de las distintas capas de una red neuronal. Cada nodo representa una neurona artificial y las conexiones entre ellas transmiten señales ponderadas.",
            "La inteligencia artificial moderna combina algoritmos sofisticados con grandes cantidades de datos para entrenar sistemas que pueden realizar tareas complejas como reconocimiento de imágenes, procesamiento de lenguaje natural y toma de decisiones.",
            "El aprendizaje profundo ha revolucionado la IA mediante el uso de redes neuronales con muchas capas. Esto permite representar conceptos abstractos y detectar patrones sutiles en los datos.",
            "Este es un proyecto demostrativo que ilustra los principios básicos del funcionamiento de las redes neuronales. Aunque simplificado, muestra el concepto de procesamiento por capas característico del deep learning."
        ]
    
    def setup_ui(self):
        # Marco principal
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = Label(main_frame, text="RED NEURONAL Benjiro v2.3", font=("Arial", 24, "bold"), bg='black', fg='#00FFFF')
        title_label.pack(pady=10)
        
        # Marco para la visualización
        self.viz_frame = tk.Frame(main_frame, bg='black')
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Marco para la entrada y salida de texto
        control_frame = tk.Frame(main_frame, bg='black')
        control_frame.pack(fill=tk.X, pady=10)
        
        # Campo de entrada
        input_frame = tk.Frame(control_frame, bg='black')
        input_frame.pack(fill=tk.X, pady=5)
        
        input_label = Label(input_frame, text="Pregunta", font=("Arial", 12), bg='black', fg='white')
        input_label.pack(side=tk.LEFT, padx=5)
        
        self.input_entry = Entry(input_frame, font=("Arial", 12), width=50, bg='#222', fg='white')
        self.input_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", lambda event: self.process_query())
        
        self.submit_btn = Button(input_frame, text="Enviar", command=self.process_query, bg='#00AAFF', fg='white', font=("Arial", 10, "bold"))
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        
        # Área de respuesta
        output_frame = tk.Frame(control_frame, bg='black')
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        output_label = Label(output_frame, text="Respuesta:", font=("Arial", 12), bg='black', fg='white')
        output_label.pack(anchor=tk.W, padx=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, width=70, font=("Arial", 11), bg='#222', fg='#00FFFF')
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
        
        # Estado de procesamiento
        self.status_label = Label(main_frame, text="Listo para responder cuestiones sobre IA", font=("Arial", 10), bg='black', fg='#AAAAAA')
        self.status_label.pack(pady=5)
    
    def setup_visualization(self):
        # Crear figura para la visualización
        self.fig = Figure(figsize=(10, 4), facecolor='black')
        self.ax = self.fig.add_subplot(111, facecolor='black')
        
        # Configuración básica del gráfico
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 6)
        self.ax.axis('off')
        
        # Crear capas de la red neuronal
        self.layers = [
            {'x': 2, 'neurons': 6, 'color': '#FF00FF'},   # Capa de entrada
            {'x': 4, 'neurons': 10, 'color': '#00FFFF'},  # Capa oculta 1 (Procesamiento 1)
            {'x': 6, 'neurons': 8, 'color': '#FFFF00'},   # Capa oculta 2 (Prosesamiento 2)
            {'x': 8, 'neurons': 4, 'color': '#00FF00'}    # Capa de salida
        ]
        
        # Crear neuronas y conexiones
        self.neurons = []
        self.connections = []
        self.activation_levels = []
        
        for layer_idx, layer in enumerate(self.layers):
            layer_neurons = []
            layer_activations = []
            
            spacing = 5 / (layer['neurons'] + 1)
            for i in range(layer['neurons']):
                y_pos = 0.5 + spacing * (i + 1)
                neuron = self.ax.scatter(layer['x'], y_pos, s=100, color=layer['color'], alpha=0.7, edgecolors='white')
                layer_neurons.append((layer['x'], y_pos))
                layer_activations.append(0.1)  # Nivel de activación inicial
            
            self.neurons.append(layer_neurons)
            self.activation_levels.append(layer_activations)
            
            # Crear conexiones con la capa anterior
            if layer_idx > 0:
                for start_x, start_y in self.neurons[layer_idx-1]:
                    for end_x, end_y in layer_neurons:
                        line, = self.ax.plot([start_x, end_x], [start_y, end_y], color='#444444', linewidth=0.5, alpha=0.3)
                        self.connections.append(line)
        
        # Añadir la figura al marco de visualización
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Iniciar animación
        self.anim = animation.FuncAnimation(self.fig, self.update_animation, frames=100, interval=50, blit=False)
        
    def update_animation(self, frame):
        # Actualización básica para mantener algo de movimiento
        if self.thinking:
            # Animación más intensa durante el "pensamiento"
            for layer_idx, layer_activations in enumerate(self.activation_levels):
                for i in range(len(layer_activations)):
                    # Mayor activación y variabilidad durante el pensamiento
                    self.activation_levels[layer_idx][i] = random.uniform(0.5, 1.0)
                    self.ax.collections[layer_idx].set_sizes([100 * self.activation_levels[layer_idx][i]])
                    self.ax.collections[layer_idx].set_alpha(0.5 + 0.5 * self.activation_levels[layer_idx][i])
            
            # Actualizar conexiones con más intensidad 
            for conn in self.connections:
                conn.set_alpha(random.uniform(0.3, 0.8))
                conn.set_color(plt.cm.viridis(random.random()))
                conn.set_linewidth(random.uniform(0.5, 2.0))
        else:
            # Animación estándar en reposo
            for layer_idx, layer_activations in enumerate(self.activation_levels):
                for i in range(len(layer_activations)):
                    # Añadir un pequeño movimiento aleatorio a cada neurona
                    self.activation_levels[layer_idx][i] = max(0.1, min(0.4, 
                                                            self.activation_levels[layer_idx][i] + 
                                                            random.uniform(-0.02, 0.02)))
                    self.ax.collections[layer_idx].set_sizes([100 * self.activation_levels[layer_idx][i]])
                    self.ax.collections[layer_idx].set_alpha(0.5 + 0.5 * self.activation_levels[layer_idx][i])
            
            # Actualizar conexiones
            for conn in self.connections:
                conn.set_alpha(random.uniform(0.1, 0.3))
                conn.set_color(plt.cm.viridis(random.random()))
                conn.set_linewidth(random.uniform(0.3, 1.0))
    
    def simulate_thinking(self, duration=3):
        # Activar modo de pensamiento
        self.thinking = True
        
        # Simular actividad neuronal intensa durante el tiempo especificado
        time.sleep(duration)
        
        # Desactivar modo de pensamiento
        self.thinking = False
    
    def find_best_response(self, query):
        # Normalizar la consulta
        query = query.lower().strip()
        
        # Buscar coincidencias en la base de conocimiento
        best_match = None
        for key in self.qa_database.keys():
            if key in query:
                best_match = key
                break
        
        if best_match:
            # Seleccionar aleatoriamente una de las respuestas disponibles para esta clave
            return random.choice(self.qa_database[best_match])
        else:
            # Si no hay coincidencia específica, devolver una respuesta general
            return random.choice(self.general_responses)
    
    def process_query(self):
        query = self.input_entry.get().strip()
        if not query:
            return
        
        self.status_label.config(text="Analisando cuestionamiento...", fg='#FFAA00')
        self.input_entry.config(state=tk.DISABLED)
        self.submit_btn.config(state=tk.DISABLED)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Pensamiento en estado de análisis...\n")
        self.output_text.config(state=tk.DISABLED)
        self.root.update()
        
        # Iniciar el procesamiento en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self.generate_response, args=(query,)).start()
    
    def generate_response(self, query):
        # Simular actividad neuronal intensa
        self.simulate_thinking(2)
        
        # Generar respuesta
        response = self.find_best_response(query)
        
        # Actualizar interfaz con la respuesta
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, response)
        self.output_text.config(state=tk.DISABLED)
        
        self.status_label.config(text="Listo para un nuevo cuestionamiento", fg='#AAAAAA')
        self.input_entry.config(state=tk.NORMAL)
        self.submit_btn.config(state=tk.NORMAL)
        self.input_entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkVisualizer(root)
    root.mainloop()