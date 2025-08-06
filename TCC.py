import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from control import tf, step_response, feedback, rlocus
from control import poles as ctrl_poles, zeros as ctrl_zeros
from scipy.signal import lsim as scipy_lsim
import tempfile
import os
import sys
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Circle
from sympy import symbols, Poly, sympify
from sympy.abc import s
import webbrowser
from PIL import Image, ImageTk

# Configuração do ícone (adicionado)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Tente importar a biblioteca FPDF2. Se não funcionar, mostre um erro.
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    messagebox.showerror("Erro de Dependência", 
                         "A biblioteca FPDF2 não foi encontrada.\n\n"
                         "Por favor, instale-a usando o comando:\n"
                         "pip install fpdf2")
    sys.exit(1)

class AdvancedControlSystemApp:
    def __init__(self, root):
        self.root = root
        self.current_font_size = 9
        self.setup_window()
        self.setup_variables()
        self.setup_styles()
        self.create_menu_bar()
        self.create_widgets()
        self.setup_shortcuts()
        self.setup_responsive_layout()
        
    def setup_window(self):
        """Configura a janela principal"""
        self.root.title("Sistema Avançado de Análise de Controladores")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        self.root.configure(bg='#f0f0f0')
        
        # Configurações para alta DPI
        self.root.tk.call('tk', 'scaling', 1.5 if self.root.winfo_fpixels('1i') > 100 else 1.0)
        
        # Tentar carregar o ícone (adicionado)
        try:
            icon_path = resource_path('icon.ico')
            self.root.iconbitmap(icon_path)
        except:
            pass  # Se não encontrar o ícone, continua sem ele
    
    def setup_responsive_layout(self):
        """Configura o layout responsivo"""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.update_font_size()
    
    def update_font_size(self):
        """Atualiza o tamanho da fonte em todos os widgets"""
        style = ttk.Style()
        style.configure('.', font=('Segoe UI', self.current_font_size))
        style.configure('Header.TLabel', font=('Segoe UI', self.current_font_size+2, 'bold'))
        style.configure('Title.TLabel', font=('Segoe UI', self.current_font_size+4, 'bold'))
        style.configure('Large.TButton', font=('Segoe UI', self.current_font_size+1, 'bold'))
        style.configure('Exit.TButton', font=('Segoe UI', self.current_font_size+1, 'bold'), foreground='white', background='#d9534f')
    
    def increase_font(self):
        """Aumenta o tamanho da fonte"""
        if self.current_font_size < 14:  # Limite máximo razoável
            self.current_font_size += 1
            self.update_font_size()
    
    def decrease_font(self):
        """Diminui o tamanho da fonte"""
        if self.current_font_size > 8:  # Limite mínimo razoável
            self.current_font_size -= 1
            self.update_font_size()
    
    def create_menu_bar(self):
        """Cria a barra de menu superior com layout mais didático"""
        menubar = tk.Menu(self.root)
        
        # Menu Arquivo
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Novo", command=self.clear_all, accelerator="Ctrl+N")
        file_menu.add_command(label="Gerar PDF", command=self.generate_pdf_report, accelerator="Ctrl+P")
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.root.quit, accelerator="Alt+F4")
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        
        # Menu Editar
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Aumentar Fonte", command=self.increase_font, accelerator="Ctrl++")
        edit_menu.add_command(label="Diminuir Fonte", command=self.decrease_font, accelerator="Ctrl+-")
        menubar.add_cascade(label="Editar", menu=edit_menu)
        
        # Menu Gráficos
        graph_menu = tk.Menu(menubar, tearoff=0)
        graph_menu.add_command(label="Gerar Gráficos", command=self.generate_plots, accelerator="Ctrl+G")
        graph_menu.add_command(label="Mostrar Polos/Zeros", command=self.show_poles_zeros, accelerator="Ctrl+Z")
        graph_menu.add_separator()
        graph_menu.add_command(label="Limpar Tudo", command=self.clear_all, accelerator="Ctrl+L")
        menubar.add_cascade(label="Gráficos", menu=graph_menu)
        
        # Menu Ajuda
        help_menu = tk.Menu(menubar, tearoff=0)
        
        # Submenu de Exemplos
        examples_menu = tk.Menu(help_menu, tearoff=0)
        examples = {
            "Sistema de 1ª Ordem": {"num": [1], "den": [1, 1]},
            "Sistema de 2ª Ordem": {"num": [1], "den": [1, 2, 1]},
            "Sistema Instável": {"num": [1], "den": [1, -1, 1]},
            "Sistema com Integrador": {"num": [1], "den": [1, 1, 0]},
            "Sistema Oscilatório": {"num": [1], "den": [1, 0, 1]}
        }
        
        for name, params in examples.items():
            examples_menu.add_command(label=name, 
                                   command=lambda n=name, p=params: self.load_example(n, p))
        
        help_menu.add_cascade(label="Exemplos Prontos", menu=examples_menu)
        help_menu.add_separator()
        help_menu.add_command(label="Tutorial", command=self.show_tutorial)
        help_menu.add_command(label="Teoria", command=self.show_theory)
        help_menu.add_separator()
        help_menu.add_command(label="Sobre", command=self.show_about)
        
        menubar.add_cascade(label="Ajuda", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def setup_variables(self):
        """Configura as variáveis de controle"""
        self.system_params = {
            'num': [1],
            'den': [1, 1],
            'input_type': 'Degrau',
            'controller_type': 'P',
            'kp': 1.0,
            'ki': 0.0,
            'kd': 0.0,
            'num_c': [1],
            'den_c': [1],
            'show_noctrl_response': True,
            'show_ctrl_response': True,
            'show_noctrl_lgr': True,
            'show_ctrl_lgr': True
        }
    
    def setup_styles(self):
        """Configura os estilos visuais com cores mais vibrantes"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores para os gráficos
        self.graph_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'background': '#f9f9f9',
            'grid': '#dddddd'
        }
        
        style.configure('.', font=('Segoe UI', 9))
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TLabel', background='#f5f5f5')
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=5)
        
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'), foreground='#2c3e50')
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#2c3e50')
        style.configure('Graph.TFrame', background='white', relief=tk.SUNKEN, borderwidth=1)
        style.configure('Red.TButton', foreground='red')
        style.configure('Large.TButton', font=('Segoe UI', 10, 'bold'), padding=8, foreground='white', background='#3498db')
        style.configure('Example.TButton', font=('Segoe UI', 9), padding=3, width=15)
        style.configure('Exit.TButton', font=('Segoe UI', 10, 'bold'), foreground='white', background='#e74c3c')
    
    def create_widgets(self):
        """Cria todos os widgets da interface"""
        self.create_main_frames()
        self.create_control_panel()
        self.create_graph_area()
        self.create_status_bar()
    
    def create_main_frames(self):
        """Cria os frames principais com layout responsivo e scrollbar"""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controle com scrollbar (25% da largura)
        control_container = ttk.Frame(self.main_frame)
        control_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Canvas com Scrollbar para o painel lateral
        control_canvas = tk.Canvas(control_container, width=350, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(
                scrollregion=control_canvas.bbox("all")
            )
        )
        
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.control_panel = scrollable_frame  # Atualize a referência
        
        # Frame de gráficos (75% da largura)
        self.graph_area = ttk.Frame(self.main_frame)
        self.graph_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def create_control_panel(self):
        """Cria o painel de controle esquerdo com layout mais didático"""
        # Frame de título
        title_frame = ttk.Frame(self.control_panel)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="CONTROLE AVANÇADO", style='Title.TLabel').pack(side=tk.LEFT)
        
        # Botão de saída destacado
        exit_btn = ttk.Button(title_frame, text="SAIR", command=self.root.quit, 
                            style='Exit.TButton', width=8)
        exit_btn.pack(side=tk.RIGHT)
        
        self.create_system_frame()
        self.create_input_frame()
        self.create_controller_frame()
        self.create_graph_selection_frame()
        self.create_action_buttons()
    
    def create_system_frame(self):
        """Frame de configuração do sistema com melhor organização"""
        sys_frame = ttk.LabelFrame(self.control_panel, text="1. CONFIGURAÇÃO DO SISTEMA", padding=10)
        sys_frame.pack(fill=tk.X, pady=5)
        
        # Numerador
        num_frame = ttk.Frame(sys_frame)
        num_frame.pack(fill=tk.X, pady=2)
        ttk.Label(num_frame, text="Numerador:").pack(side=tk.LEFT, padx=(0, 5))
        self.num_entry = ttk.Entry(num_frame)
        self.num_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.num_entry.insert(0, "1")
        self.num_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ex: 1 2 3 para [1, 2, 3]"))
        
        # Denominador
        den_frame = ttk.Frame(sys_frame)
        den_frame.pack(fill=tk.X, pady=2)
        ttk.Label(den_frame, text="Denominador:").pack(side=tk.LEFT, padx=(0, 5))
        self.den_entry = ttk.Entry(den_frame)
        self.den_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.den_entry.insert(0, "1 1")
        self.den_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ex: 1 5 6 para [1, 5, 6]"))
        
        # Botão para análise de estabilidade
        ttk.Button(sys_frame, text="Análise de Estabilidade", command=self.analyze_stability,
                  style='Large.TButton').pack(fill=tk.X, pady=(10, 0))
    
    def analyze_stability(self):
        """Realiza a análise de estabilidade usando o critério de Routh-Hurwitz"""
        try:
            den = self.parse_coefficients(self.den_entry.get())
            if den is None:
                return
                
            stability_window = tk.Toplevel(self.root)
            stability_window.title("Análise de Estabilidade - Critério de Routh-Hurwitz")
            stability_window.geometry("900x700")
            
            # Frame principal com panedwindow para dividir tabela e gráfico
            main_paned = ttk.PanedWindow(stability_window, orient=tk.VERTICAL)
            main_paned.pack(fill=tk.BOTH, expand=True)
            
            # Frame superior para a tabela
            table_frame = ttk.Frame(main_paned)
            main_paned.add(table_frame, weight=1)
            
            # Frame inferior para o gráfico
            graph_frame = ttk.Frame(main_paned)
            main_paned.add(graph_frame, weight=1)
            
            # Criação do polinômio e cálculo da tabela
            s_poly = self.create_polynomial(den)
            rh_table, stability = self.routh_hurwitz(s_poly)
            
            # Configuração do texto explicativo
            explanation = """CRITÉRIO DE ROUTH-HURWITZ:
1. Todos os coeficientes devem ter o mesmo sinal
2. Todos os elementos da 1ª coluna devem ter o mesmo sinal
3. Mudanças de sinal na 1ª coluna indicam polos no semiplano direito"""
            
            ttk.Label(table_frame, text=explanation, wraplength=850, justify='left').pack(pady=5)
            
            # Resultado da análise
            if stability == "Estável":
                color = "#2ecc71"  # Verde
            elif stability == "Marginalmente Estável":
                color = "#f39c12"  # Amarelo
            else:
                color = "#e74c3c"  # Vermelho
                
            result_frame = ttk.Frame(table_frame)
            result_frame.pack(pady=5)
            ttk.Label(result_frame, text="Resultado: ", font=('Segoe UI', 12, 'bold')).pack(side=tk.LEFT)
            ttk.Label(result_frame, text=stability, font=('Segoe UI', 12, 'bold'), foreground=color).pack(side=tk.LEFT)
            
            # Frame para a tabela com barra de rolagem
            table_container = ttk.Frame(table_frame)
            table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Canvas e scrollbar
            canvas = tk.Canvas(table_container)
            scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.get_tk_widget().configure(
                    scrollregion=canvas.get_tk_widget().bbox("all")
                )
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Adicionar a tabela formatada
            for i, row in enumerate(rh_table):
                row_frame = ttk.Frame(scrollable_frame)
                row_frame.pack(fill=tk.X, padx=5, pady=1)
                
                # Label para a potência de s (S0, S1, etc.)
                ttk.Label(row_frame, text=row[0], width=4, anchor="e", 
                        font=('Consolas', 10)).pack(side=tk.LEFT, padx=2)
                
                # Elementos da linha
                for j, item in enumerate(row[1:]):
                    bg_color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
                    if j == 0:  # Primeira coluna (mais importante)
                        try:
                            if float(item) >= 0:
                                bg_color = "#d4edda"  # Verde claro
                            else:
                                bg_color = "#f8d7da"  # Vermelho claro
                        except ValueError:
                            pass
                            
                    ttk.Label(row_frame, text=item, width=12, relief="ridge", 
                            background=bg_color, anchor="center",
                            font=('Consolas', 10)).pack(side=tk.LEFT, padx=1)
            
            # Adicionar gráfico de polos
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Configurações do gráfico com cores personalizadas
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.axvspan(0.001, ax.get_xlim()[1], color='#f8d7da', alpha=0.3)  # Área instável
            
            G = tf(self.parse_coefficients(self.num_entry.get()), den)
            poles = ctrl_poles(G)
            
            if poles.size > 0:
                # Colorir polos de acordo com a estabilidade
                for pole in poles:
                    if pole.real < 0:
                        ax.plot(pole.real, pole.imag, 'x', markersize=10, markeredgewidth=2, 
                              color='#28a745', label='Polo Estável' if pole == poles[0] else "")
                    else:
                        ax.plot(pole.real, pole.imag, 'x', markersize=10, markeredgewidth=2, 
                              color='#dc3545', label='Polo Instável' if pole == poles[0] else "")
            
            ax.set_title('Diagrama de Polos', pad=20, fontsize=12)
            ax.set_xlabel('Parte Real', fontsize=10)
            ax.set_ylabel('Parte Imaginária', fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, color=self.graph_colors['grid'])
            ax.set_facecolor(self.graph_colors['background'])
            
            if poles.size > 0:
                ax.legend()
            
            # Círculo unitário para referência
            circle = Circle((0, 0), 1, fill=False, color='#6c757d', linestyle='--', alpha=0.3)
            ax.add_patch(circle)
            
            # Adiciona o gráfico ao frame inferior
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Adiciona barra de ferramentas para o gráfico
            toolbar = NavigationToolbar2Tk(canvas, graph_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na análise de estabilidade: {str(e)}")

    def create_polynomial(self, coefficients):
        """Cria um polinômio sympy a partir dos coeficientes"""
        s_poly = sum(coef * s**i for i, coef in enumerate(reversed(coefficients)))
        return Poly(s_poly, s)
    
    def routh_hurwitz(self, s_poly):
        """Implementação corrigida do critério de Routh-Hurwitz"""
        coeffs = s_poly.all_coeffs()
        n = len(coeffs)
        
        # Verifica se todos os coeficientes são positivos ou negativos
        if not (all(c > 0 for c in coeffs) or all(c < 0 for c in coeffs)):
            return [["s⁰"] + [str(c) for c in coeffs[::2]]], "Instável (coeficientes com sinais diferentes)"
        
        # Inicializa a tabela
        table = []
        first_row = ["s⁰"] + [str(c) for c in coeffs[::2]]
        table.append(first_row)
        
        if n > 1:
            second_row = ["s¹"] + [str(c) for c in coeffs[1::2]]
            if n % 2 == 1:
                second_row.append("0")  # Completa com zero se necessário
            table.append(second_row)
        
        # Preenche as linhas restantes
        for i in range(2, n):
            prev_row = table[i-1]
            prev_prev_row = table[i-2]
            
            # Verifica se o primeiro elemento da linha anterior é zero
            if float(prev_row[1]) == 0:
                return table, "Instável (divisão por zero na tabela)"
            
            current_row = [f"s^{i}"] if i > 1 else [f"s{i}"]
            
            for j in range(1, len(prev_prev_row)-1):
                try:
                    a = (float(prev_row[1]) * float(prev_prev_row[j+1]) - 
                        float(prev_prev_row[1]) * float(prev_row[j+1])) / float(prev_row[1])
                except:
                    a = 0
                current_row.append(f"{a:.4f}")
            
            table.append(current_row)
            
            # Remove zeros extras no final
            while len(table[i]) > 1 and table[i][-1] == "0":
                table[i] = table[i][:-1]
            
            if len(table[i]) <= 1:
                break
        
        # Verifica estabilidade
        first_col = [float(row[1]) for row in table if len(row) > 1]
        sign_changes = sum(1 for i in range(len(first_col)-1) if first_col[i]*first_col[i+1] < 0)
        
        if sign_changes > 0:
            stability = f"Instável ({sign_changes} polos no semiplano direito)"
        else:
            if any(abs(float(row[1])) < 1e-10 for row in table if len(row) > 1):
                stability = "Marginalmente Estável (polos no eixo imaginário)"
            else:
                stability = "Estável"
        
        return table, stability
    
    def show_tooltip(self, message):
        """Mostra uma dica contextual"""
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{self.root.winfo_pointerx()+10}+{self.root.winfo_pointery()+10}")
        label = ttk.Label(tooltip, text=message, background="#ffffe0", relief="solid", borderwidth=1,
                         font=('Segoe UI', 9))
        label.pack()
        self.root.after(3000, tooltip.destroy)
    
    def create_input_frame(self):
        """Frame de seleção de tipo de entrada com layout melhorado"""
        input_frame = ttk.LabelFrame(self.control_panel, text="2. TIPO DE ENTRADA", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_var = tk.StringVar(value="Degrau")
        
        # Frame para os botões de rádio
        radio_frame = ttk.Frame(input_frame)
        radio_frame.pack(fill=tk.X, pady=5)
        
        inputs = ["Degrau", "Rampa", "Parábola"]
        for i, inp in enumerate(inputs):
            rb = ttk.Radiobutton(radio_frame, text=inp, variable=self.input_var, value=inp,
                                style='Toolbutton')
            rb.pack(side=tk.LEFT, expand=True, padx=2)
    
    def create_controller_frame(self):
        """Frame de configuração do controlador com layout mais intuitivo"""
        ctrl_frame = ttk.LabelFrame(self.control_panel, text="3. CONFIGURAÇÃO DO CONTROLADOR", padding=10)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        # Tipo de controlador
        type_frame = ttk.Frame(ctrl_frame)
        type_frame.pack(fill=tk.X, pady=2)
        ttk.Label(type_frame, text="Tipo:").pack(side=tk.LEFT, padx=(0, 5))
        self.ctrl_type = ttk.Combobox(type_frame, values=["P", "PI", "PD", "PID", "Avanço", "Atraso", "Avanço-Atraso"])
        self.ctrl_type.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ctrl_type.set("P")
        self.ctrl_type.bind("<<ComboboxSelected>>", self.update_controller_fields)
        
        # Frame para parâmetros do controlador
        self.ctrl_params_frame = ttk.Frame(ctrl_frame)
        self.ctrl_params_frame.pack(fill=tk.X, pady=(5, 0))
        self.update_controller_fields()
    
    def create_graph_selection_frame(self):
        """Frame de seleção de gráficos com layout mais organizado"""
        graph_select_frame = ttk.LabelFrame(self.control_panel, text="4. GRÁFICOS A EXIBIR", padding=10)
        graph_select_frame.pack(fill=tk.X, pady=5)
        
        # Organiza em duas colunas
        left_frame = ttk.Frame(graph_select_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(graph_select_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Checkboxes para resposta temporal
        ttk.Label(left_frame, text="Resposta Temporal:", font=('Segoe UI', 9, 'bold')).pack(anchor="w", pady=(0, 5))
        self.show_noctrl_resp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Sem controlador", 
                       variable=self.show_noctrl_resp_var).pack(anchor="w", pady=2)
        
        self.show_ctrl_resp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Com controlador", 
                       variable=self.show_ctrl_resp_var).pack(anchor="w", pady=2)
        
        # Checkboxes para LGR
        ttk.Label(right_frame, text="Lugar Geométrico das Raízes:", font=('Segoe UI', 9, 'bold')).pack(anchor="w", pady=(0, 5))
        self.show_noctrl_lgr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, text="Sem controlador", 
                       variable=self.show_noctrl_lgr_var).pack(anchor="w", pady=2)
        
        self.show_ctrl_lgr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, text="Com controlador", 
                       variable=self.show_ctrl_lgr_var).pack(anchor="w", pady=2)
    
    def load_example(self, example_name, example_params):
        """Carrega um exemplo pré-definido"""
        self.num_entry.delete(0, tk.END)
        self.num_entry.insert(0, " ".join(map(str, example_params["num"])))
        self.den_entry.delete(0, tk.END)
        self.den_entry.insert(0, " ".join(map(str, example_params["den"])))
        self.status_label.config(text=f"Exemplo carregado: {example_name}")
    
    def create_action_buttons(self):
        """Cria os botões de ação com layout mais intuitivo"""
        btn_frame = ttk.Frame(self.control_panel)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Botão principal para gerar gráficos
        ttk.Button(btn_frame, text="Gerar Gráficos", command=self.generate_plots, 
                style='Large.TButton').pack(fill=tk.X, pady=2)
        
        # Frame para botões secundários
        secondary_btn_frame = ttk.Frame(btn_frame)
        secondary_btn_frame.pack(fill=tk.X, pady=5)
        
        # Botões secundários organizados em grade
        ttk.Button(secondary_btn_frame, text="Polos/Zeros", command=self.show_poles_zeros,
                style='Large.TButton').grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        ttk.Button(secondary_btn_frame, text="Análise Estabilidade", command=self.analyze_stability,
                style='Large.TButton').grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        ttk.Button(secondary_btn_frame, text="Gerar PDF", command=self.generate_pdf_report,
                style='Large.TButton').grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        
        ttk.Button(secondary_btn_frame, text="Limpar Tudo", command=self.clear_all, 
                style='Large.TButton').grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        
        # Configurar pesos das colunas para expansão
        secondary_btn_frame.columnconfigure(0, weight=1)
        secondary_btn_frame.columnconfigure(1, weight=1)
    
    def create_graph_area(self):
        """Cria a área de exibição dos gráficos com layout responsivo e cores"""
        self.graph_notebook = ttk.Notebook(self.graph_area)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Abas
        self.create_response_tab()
        self.create_lgr_tab()
        self.create_pz_tab()
        self.create_stability_tab()
    
    def create_response_tab(self):
        """Aba de resposta temporal corrigida"""
        self.resp_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.resp_frame, text="Resposta Temporal")
        
        # Frame contêiner para os dois gráficos
        resp_container = ttk.Frame(self.resp_frame)
        resp_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Gráfico sem controlador - CORREÇÃO AQUI
        self.noctrl_resp_frame = ttk.LabelFrame(resp_container, text="Sem Controlador", padding=5)
        self.noctrl_resp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_noctrl_resp = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax_noctrl_resp = self.fig_noctrl_resp.add_subplot(111)
        
        # Container para canvas e toolbar
        canvas_frame = ttk.Frame(self.noctrl_resp_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_noctrl_resp = FigureCanvasTkAgg(self.fig_noctrl_resp, master=canvas_frame)
        self.canvas_noctrl_resp.draw()
        self.canvas_noctrl_resp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barra de ferramentas em frame separado
        toolbar_frame = ttk.Frame(self.noctrl_resp_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas_noctrl_resp, toolbar_frame)
        toolbar.update()
        
        # Repetir a mesma estrutura para o gráfico com controlador
        self.ctrl_resp_frame = ttk.LabelFrame(resp_container, text="Com Controlador", padding=5)
        self.ctrl_resp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_ctrl_resp = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax_ctrl_resp = self.fig_ctrl_resp.add_subplot(111)
        
        canvas_frame = ttk.Frame(self.ctrl_resp_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_ctrl_resp = FigureCanvasTkAgg(self.fig_ctrl_resp, master=canvas_frame)
        self.canvas_ctrl_resp.draw()
        self.canvas_ctrl_resp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(self.ctrl_resp_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas_ctrl_resp, toolbar_frame)
        toolbar.update()
    
    def create_lgr_tab(self):
        """Aba de LGR com layout melhorado e cores"""
        self.lgr_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.lgr_frame, text="Lugar Geométrico das Raízes")
        
        # Frame contêiner para os dois gráficos
        lgr_container = ttk.Frame(self.lgr_frame)
        lgr_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Gráfico sem controlador
        self.noctrl_lgr_frame = ttk.LabelFrame(lgr_container, text="Sem Controlador", padding=5,
                                             style='Graph.TFrame')
        self.noctrl_lgr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_noctrl_lgr = plt.Figure(figsize=(6, 4), dpi=100, facecolor=self.graph_colors['background'])
        self.ax_noctrl_lgr = self.fig_noctrl_lgr.add_subplot(111)
        self.setup_plot_style(self.ax_noctrl_lgr)
        self.canvas_noctrl_lgr = FigureCanvasTkAgg(self.fig_noctrl_lgr, master=self.noctrl_lgr_frame)
        
        # Adicionar barra de ferramentas interativa
        toolbar = NavigationToolbar2Tk(self.canvas_noctrl_lgr, self.noctrl_lgr_frame)
        toolbar.update()
        self.canvas_noctrl_lgr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico com controlador
        self.ctrl_lgr_frame = ttk.LabelFrame(lgr_container, text="Com Controlador", padding=5,
                                           style='Graph.TFrame')
        self.ctrl_lgr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_ctrl_lgr = plt.Figure(figsize=(6, 4), dpi=100, facecolor=self.graph_colors['background'])
        self.ax_ctrl_lgr = self.fig_ctrl_lgr.add_subplot(111)
        self.setup_plot_style(self.ax_ctrl_lgr)
        self.canvas_ctrl_lgr = FigureCanvasTkAgg(self.fig_ctrl_lgr, master=self.ctrl_lgr_frame)
        
        # Adicionar barra de ferramentas interativa
        toolbar = NavigationToolbar2Tk(self.canvas_ctrl_lgr, self.ctrl_lgr_frame)
        toolbar.update()
        self.canvas_ctrl_lgr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_plot_style(self, ax):
        """Configura o estilo dos gráficos para melhor visualização com cores"""
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, color=self.graph_colors['grid'])
        ax.set_facecolor(self.graph_colors['background'])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=3)
        
        # Configurações de cor para os eixos e título
        ax.spines['bottom'].set_color(self.graph_colors['primary'])
        ax.spines['top'].set_color(self.graph_colors['primary']) 
        ax.spines['right'].set_color(self.graph_colors['primary'])
        ax.spines['left'].set_color(self.graph_colors['primary'])
        
        ax.title.set_color(self.graph_colors['primary'])
        ax.xaxis.label.set_color(self.graph_colors['primary'])
        ax.yaxis.label.set_color(self.graph_colors['primary'])
        
        ax.tick_params(axis='x', colors=self.graph_colors['primary'])
        ax.tick_params(axis='y', colors=self.graph_colors['primary'])
    
    def create_pz_tab(self):
        """Aba de polos e zeros com layout melhorado e cores"""
        self.pz_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.pz_frame, text="Polos e Zeros")
        
        # Frame principal
        pz_container = ttk.Frame(self.pz_frame)
        pz_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gráfico
        graph_frame = ttk.LabelFrame(pz_container, text="Diagrama de Polos e Zeros", padding=5,
                                   style='Graph.TFrame')
        graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.fig_pz = plt.Figure(figsize=(8, 6), dpi=100, facecolor=self.graph_colors['background'])
        self.ax_pz = self.fig_pz.add_subplot(111)
        self.setup_plot_style(self.ax_pz)
        self.canvas_pz = FigureCanvasTkAgg(self.fig_pz, master=graph_frame)
        
        # Adicionar barra de ferramentas interativa
        toolbar = NavigationToolbar2Tk(self.canvas_pz, graph_frame)
        toolbar.update()
        self.canvas_pz.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Área de informações textuais
        info_frame = ttk.LabelFrame(pz_container, text="Informações", padding=5)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.pz_info = tk.Text(info_frame, wrap=tk.WORD, height=10, font=('Consolas', 10),
                             bg='white', fg='#333333')
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.pz_info.yview)
        self.pz_info.configure(yscrollcommand=scrollbar.set)
        
        self.pz_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pz_info.insert(tk.END, "Informações sobre polos e zeros serão exibidas aqui...")
        self.pz_info.config(state=tk.DISABLED)
    
    def create_stability_tab(self):
        """Aba de análise de estabilidade com layout melhorado"""
        self.stability_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.stability_frame, text="Análise de Estabilidade")
        
        # Frame principal
        stability_container = ttk.Frame(self.stability_frame)
        stability_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Texto explicativo
        explanation = """ANÁLISE DE ESTABILIDADE

O critério de Routh-Hurwitz permite determinar a estabilidade de um sistema linear 
sem calcular explicitamente os polos da função de transferência.

Para usar:
1. Configure os coeficientes do denominador na aba principal
2. Clique em "Análise de Estabilidade" no painel esquerdo

O sistema será classificado como:
- Estável: Todos os polos no semiplano esquerdo
- Marginalmente Estável: Polos no eixo imaginário
- Instável: Pelo menos um polo no semiplano direito"""
        
        ttk.Label(stability_container, text=explanation, wraplength=700, justify='left').pack(pady=10)
        
        # Botão para análise
        ttk.Button(stability_container, text="Realizar Análise de Estabilidade", 
                  command=self.analyze_stability, style='Large.TButton').pack(pady=10)
    
    def show_tutorial(self):
        """Mostra um tutorial interativo com layout melhorado"""
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("Tutorial Interativo")
        tutorial_window.geometry("800x600")
        
        notebook = ttk.Notebook(tutorial_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Página 1: Introdução
        intro_frame = ttk.Frame(notebook)
        notebook.add(intro_frame, text="Introdução")
        
        intro_text = """BEM-VINDO AO TUTORIAL INTERATIVO

Este software foi desenvolvido para ajudar no estudo de sistemas de controle.

Principais recursos:
1. Visualização da resposta temporal do sistema
2. Análise do Lugar Geométrico das Raízes (LGR)
3. Diagrama de Polos e Zeros
4. Análise de estabilidade pelo critério de Routh-Hurwitz
5. Geração de relatórios em PDF

Como usar:
1. Configure o sistema (numerador e denominador)
2. Selecione o tipo de entrada
3. Escolha e ajuste o controlador
4. Selecione os gráficos desejados
5. Clique em "Gerar Gráficos"

Experimente os exemplos prontos no menu Ajuda!"""
        
        ttk.Label(intro_frame, text=intro_text, wraplength=750, justify='left').pack(pady=20)
        
        # Página 2: Passo a Passo
        steps_frame = ttk.Frame(notebook)
        notebook.add(steps_frame, text="Passo a Passo")
        
        steps = [
            ("1. Configuração do Sistema", "Defina os coeficientes do numerador e denominador da função de transferência."),
            ("2. Tipo de Entrada", "Selecione o tipo de entrada (degrau, rampa ou parábola) para análise."),
            ("3. Controlador", "Escolha o tipo de controlador e ajuste seus parâmetros."),
            ("4. Gráficos", "Selecione quais gráficos deseja visualizar."),
            ("5. Análise", "Gere os gráficos e analise o comportamento do sistema."),
            ("6. Exportação", "Gere um relatório em PDF com os resultados.")
        ]
        
        for i, (title, desc) in enumerate(steps):
            step_frame = ttk.Frame(steps_frame)
            step_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(step_frame, text=title, font=('Segoe UI', 10, 'bold'), 
                     foreground=self.graph_colors['primary']).pack(anchor="w")
            ttk.Label(step_frame, text=desc, wraplength=750, justify='left').pack(anchor="w")
    
    def show_theory(self):
        """Mostra a teoria básica dos controladores com layout melhorado"""
        theory_window = tk.Toplevel(self.root)
        theory_window.title("Teoria dos Controladores")
        theory_window.geometry("900x700")
        
        notebook = ttk.Notebook(theory_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Página 1: Tipos de Controladores
        ctrl_frame = ttk.Frame(notebook)
        notebook.add(ctrl_frame, text="Tipos de Controladores")
        
        ctrl_text = """TIPOS DE CONTROLADORES

1. Controlador Proporcional (P)
   - Ação: u(t) = Kp * e(t)
   - Efeito: Reduz o erro em regime permanente, mas pode causar oscilações

2. Controlador Proporcional-Integral (PI)
   - Ação: u(t) = Kp*e(t) + Ki*∫e(t)dt
   - Efeito: Elimina erro em regime permanente, mas pode reduzir a estabilidade

3. Controlador Proporcional-Derivativo (PD)
   - Ação: u(t) = Kp*e(t) + Kd*de(t)/dt
   - Efeito: Melhora a estabilidade e resposta transitória

4. Controlador PID
   - Combina as ações P, I e D
   - Oferece bons resultados para diversos sistemas"""
        
        text_widget = tk.Text(ctrl_frame, wrap=tk.WORD, padx=10, pady=10, font=('Segoe UI', 10))
        text_widget.insert(tk.END, ctrl_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Página 2: Estabilidade
        stability_frame = ttk.Frame(notebook)
        notebook.add(stability_frame, text="Estabilidade")
        
        stability_text = """CRITÉRIO DE ESTABILIDADE

Um sistema é estável se todos os polos de sua função de transferência possuem parte real negativa.

Critério de Routh-Hurwitz:
- Método para determinar a estabilidade sem calcular os polos
- Analisa os coeficientes do polinômio característico
- Verifica se há mudanças de sinal na primeira coluna do array de Routh

Classificação:
- Estável: Todos os polos no semiplano esquerdo
- Marginalmente Estável: Polos no eixo imaginário
- Instável: Pelo menos um polo no semiplano direito"""
        
        text_widget = tk.Text(stability_frame, wrap=tk.WORD, padx=10, pady=10, font=('Segoe UI', 10))
        text_widget.insert(tk.END, stability_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Página 3: LGR
        lgr_frame = ttk.Frame(notebook)
        notebook.add(lgr_frame, text="Lugar Geométrico das Raízes")
        
        lgr_text = """LUGAR GEOMÉTRICO DAS RAÍZES (LGR)

O LGR mostra como os polos do sistema em malha fechada variam com o ganho K.

Propriedades:
1. O LGR começa nos polos de malha aberta (K=0) e termina nos zeros (K→∞)
2. O número de ramos é igual ao número de polos
3. Os ramos são simétricos em relação ao eixo real
4. Regiões do eixo real pertencem ao LGR se houver um número ímpar de polos+zeros à direita

Uso:
- Projeto de controladores
- Análise de estabilidade
- Determinação de faixas de ganho estáveis"""
        
        text_widget = tk.Text(lgr_frame, wrap=tk.WORD, padx=10, pady=10, font=('Segoe UI', 10))
        text_widget.insert(tk.END, lgr_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Adiciona botão para abrir material complementar
        def open_material():
            webbrowser.open("https://www.youtube.com/watch?v=Z3I7Fz7gAUw")
        
        btn_frame = ttk.Frame(theory_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Abrir Vídeo Aula Complementar", 
                 command=open_material, style='Large.TButton').pack(fill=tk.X)
    
    def show_about(self):
        """Mostra informações sobre o software com layout melhorado"""
        about_window = tk.Toplevel(self.root)
        about_window.title("Sobre")
        about_window.geometry("400x300")
        
        # Frame principal
        main_frame = ttk.Frame(about_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        ttk.Label(main_frame, text="Sistema Avançado de Análise de Controladores", 
                 font=('Segoe UI', 12, 'bold'), foreground=self.graph_colors['primary']).pack(pady=10)
        
        # Versão
        ttk.Label(main_frame, text="Versão 2.0", font=('Segoe UI', 10)).pack()
        
        # Descrição
        ttk.Label(main_frame, text="Desenvolvido para auxiliar no aprendizado\n de sistemas de controle", 
                 wraplength=350, justify='center').pack(pady=10)
        
        # Recursos
        ttk.Label(main_frame, text="Recursos:", font=('Segoe UI', 10, 'bold')).pack()
        
        features_frame = ttk.Frame(main_frame)
        features_frame.pack()
        
        features = [
            "- Análise de resposta temporal",
            "- Lugar Geométrico das Raízes",
            "- Diagrama de Polos e Zeros",
            "- Análise de estabilidade"
        ]
        
        for feature in features:
            ttk.Label(features_frame, text=feature).pack(anchor="w")
        
        # Botão de fechar
        ttk.Button(main_frame, text="Fechar", command=about_window.destroy,
                  style='Large.TButton').pack(pady=10)
    
    def create_status_bar(self):
        """Cria a barra de status com informações úteis"""
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Pronto. Use Ctrl+G para gerar gráficos.")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        help_btn = ttk.Button(self.status_bar, text="Ajuda", command=self.show_help,
                             style='Small.TButton')
        help_btn.pack(side=tk.RIGHT, padx=5)
    
    def show_help(self):
        """Mostra uma janela de ajuda com layout melhorado"""
        help_text = """INSTRUÇÕES DE USO:

1. Configure os parâmetros do sistema (numerador e denominador)
2. Selecione o tipo de entrada (degrau, rampa ou parábola)
3. Escolha o tipo de controlador e ajuste seus parâmetros
4. Selecione quais gráficos deseja visualizar
5. Clique em "Gerar Gráficos" ou pressione Ctrl+G

Atalhos:
Ctrl+G - Gerar gráficos
Ctrl+P - Mostrar polos/zeros
Ctrl+R - Gerar relatório PDF
Ctrl+L - Limpar tudo
F1 - Mostrar esta ajuda"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Ajuda")
        help_window.geometry("500x400")
        
        text = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10, font=('Segoe UI', 10),
                      bg='white', fg='#333333')
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)
        
        close_btn = ttk.Button(help_window, text="Fechar", command=help_window.destroy,
                             style='Large.TButton')
        close_btn.pack(pady=10)
    
    def update_controller_fields(self, event=None):
        """Atualiza os campos do controlador com layout mais organizado"""
        for widget in self.ctrl_params_frame.winfo_children():
            widget.destroy()
        
        ctrl_type = self.ctrl_type.get()
        row = 0
        
        # Kp (comum a todos)
        kp_frame = ttk.Frame(self.ctrl_params_frame)
        kp_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(kp_frame, text="Kp:").pack(side=tk.LEFT, padx=(0, 5))
        self.kp_entry = ttk.Entry(kp_frame, width=10)
        self.kp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.kp_entry.insert(0, "1.0")
        self.kp_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ganho proporcional"))
        row += 1
        
        # Campos específicos
        if ctrl_type in ["PI", "PID"]:
            ki_frame = ttk.Frame(self.ctrl_params_frame)
            ki_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Label(ki_frame, text="Ki:").pack(side=tk.LEFT, padx=(0, 5))
            self.ki_entry = ttk.Entry(ki_frame, width=10)
            self.ki_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.ki_entry.insert(0, "0.1")
            self.ki_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ganho integral"))
            row += 1
        
        if ctrl_type in ["PD", "PID"]:
            kd_frame = ttk.Frame(self.ctrl_params_frame)
            kd_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Label(kd_frame, text="Kd:").pack(side=tk.LEFT, padx=(0, 5))
            self.kd_entry = ttk.Entry(kd_frame, width=10)
            self.kd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.kd_entry.insert(0, "0.1")
            self.kd_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ganho derivativo"))
            row += 1
        
        if ctrl_type in ["Avanço", "Atraso", "Avanço-Atraso"]:
            num_c_frame = ttk.Frame(self.ctrl_params_frame)
            num_c_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Label(num_c_frame, text="Numerador:").pack(side=tk.LEFT, padx=(0, 5))
            self.num_c_entry = ttk.Entry(num_c_frame)
            self.num_c_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.num_c_entry.insert(0, "1")
            self.num_c_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do numerador"))
            row += 1
            
            den_c_frame = ttk.Frame(self.ctrl_params_frame)
            den_c_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Label(den_c_frame, text="Denominador:").pack(side=tk.LEFT, padx=(0, 5))
            self.den_c_entry = ttk.Entry(den_c_frame)
            self.den_c_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.den_c_entry.insert(0, "1 1")
            self.den_c_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do denominador"))
            row += 1
            
            if ctrl_type == "Avanço-Atraso":
                num_at_frame = ttk.Frame(self.ctrl_params_frame)
                num_at_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
                ttk.Label(num_at_frame, text="Numerador Atraso:").pack(side=tk.LEFT, padx=(0, 5))
                self.num_at_entry = ttk.Entry(num_at_frame)
                self.num_at_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.num_at_entry.insert(0, "1")
                self.num_at_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do numerador (atraso)"))
                row += 1
                
                den_at_frame = ttk.Frame(self.ctrl_params_frame)
                den_at_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
                ttk.Label(den_at_frame, text="Denominador Atraso:").pack(side=tk.LEFT, padx=(0, 5))
                self.den_at_entry = ttk.Entry(den_at_frame)
                self.den_at_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.den_at_entry.insert(0, "1 1")
                self.den_at_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do denominador (atraso)"))
                row += 1
    
    def parse_coefficients(self, entry_text):
        """Converte texto para lista de coeficientes"""
        try:
            clean_text = entry_text.strip("[]")
            return [float(x.strip()) for x in clean_text.split()]
        except ValueError:
            messagebox.showerror("Erro", "Formato inválido para coeficientes. Use números separados por espaços.")
            return None
    
    def get_controller_tf(self):
        """Retorna a função de transferência do controlador"""
        try:
            ctrl_type = self.ctrl_type.get()
            kp = float(self.kp_entry.get())
            
            if ctrl_type == "P":
                return tf([kp], [1])
            
            elif ctrl_type == "PI":
                ki = float(self.ki_entry.get())
                return tf([kp, ki], [1, 0])
            
            elif ctrl_type == "PD":
                kd = float(self.kd_entry.get())
                return tf([kd, kp], [1])
            
            elif ctrl_type == "PID":
                ki = float(self.ki_entry.get())
                kd = float(self.kd_entry.get())
                return tf([kd, kp, ki], [1, 0])
            
            elif ctrl_type in ["Avanço", "Atraso"]:
                num_c = self.parse_coefficients(self.num_c_entry.get())
                den_c = self.parse_coefficients(self.den_c_entry.get())
                if num_c is None or den_c is None:
                    return None
                return tf(np.array(num_c) * kp, np.array(den_c))
            
            elif ctrl_type == "Avanço-Atraso":
                num_c = self.parse_coefficients(self.num_c_entry.get())
                den_c = self.parse_coefficients(self.den_c_entry.get())
                num_at = self.parse_coefficients(self.num_at_entry.get())
                den_at = self.parse_coefficients(self.den_at_entry.get())
                
                if None in [num_c, den_c, num_at, den_at]:
                    return None
                    
                G_avanco = tf(np.array(num_c), np.array(den_c))
                G_atraso = tf(np.array(num_at), np.array(den_at))
                return kp * G_avanco * G_atraso
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Valor inválido: {str(e)}")
            return None
    
    def generate_plots(self):
        """Gera todos os gráficos selecionados com cores personalizadas"""
        try:
            # Obter parâmetros do sistema
            num = self.parse_coefficients(self.num_entry.get())
            den = self.parse_coefficients(self.den_entry.get())
            
            if num is None or den is None:
                return
                
            G = tf(num, den)
            
            # Limpar gráficos antes de gerar novos para evitar sobreposição
            self.ax_noctrl_resp.clear()
            self.ax_ctrl_resp.clear()
            self.ax_noctrl_lgr.clear()
            self.ax_ctrl_lgr.clear()

            # Atualizar gráficos sem controlador
            if self.show_noctrl_resp_var.get() or self.show_noctrl_lgr_var.get():
                self.update_noctrl_plots(G)
            
            # Obter controlador e atualizar gráficos com controlador
            if self.show_ctrl_resp_var.get() or self.show_ctrl_lgr_var.get():
                Gc = self.get_controller_tf()
                if Gc is not None:
                    self.update_ctrl_plots(G, Gc)
            
            # Atualizar gráfico de polos e zeros
            self.update_pz_plot(G)
            
            self.status_label.config(text="Gráficos gerados com sucesso!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao gerar gráficos: {str(e)}")
            self.status_label.config(text="Erro ao gerar gráficos")
    
    def update_noctrl_plots(self, G):
        """Atualiza gráficos sem controlador com estilo melhorado e cores"""
        input_type = self.input_var.get()
        
        if self.show_noctrl_resp_var.get():
            self.ax_noctrl_resp.clear()
            
            if input_type == "Degrau":
                t, y = step_response(feedback(G, 1))
                self.ax_noctrl_resp.plot(t, y, linewidth=2, color=self.graph_colors['primary'])
                self.ax_noctrl_resp.set_title('Resposta ao Degrau - Sem Controlador', pad=20, 
                                            fontsize=12, color=self.graph_colors['primary'])
            elif input_type == "Rampa":
                t = np.linspace(0, 10, 1000)
                u = t
                sys_fb = feedback(G, 1)
                t, y, _ = scipy_lsim(sys_fb, u, t)
                self.ax_noctrl_resp.plot(t, y, linewidth=2, color=self.graph_colors['primary'], label='Saída')
                self.ax_noctrl_resp.plot(t, u, '--', linewidth=2, color=self.graph_colors['secondary'], label='Entrada')
                self.ax_noctrl_resp.set_title('Resposta à Rampa - Sem Controlador', pad=20, 
                                            fontsize=12, color=self.graph_colors['primary'])
                self.ax_noctrl_resp.legend()
            elif input_type == "Parábola":
                t = np.linspace(0, 10, 1000)
                u = (t**2) / 2
                sys_fb = feedback(G, 1)
                t, y, _ = scipy_lsim(sys_fb, u, t)
                self.ax_noctrl_resp.plot(t, y, linewidth=2, color=self.graph_colors['primary'], label='Saída')
                self.ax_noctrl_resp.plot(t, u, '--', linewidth=2, color=self.graph_colors['secondary'], label='Entrada')
                self.ax_noctrl_resp.set_title('Resposta à Parábola - Sem Controlador', pad=20, 
                                            fontsize=12, color=self.graph_colors['primary'])
                self.ax_noctrl_resp.legend()
            
            self.setup_plot_style(self.ax_noctrl_resp)
            self.ax_noctrl_resp.set_xlabel('Tempo (s)', fontsize=10)
            self.ax_noctrl_resp.set_ylabel('Saída', fontsize=10)
            self.canvas_noctrl_resp.draw()
        
        if self.show_noctrl_lgr_var.get():
            self.ax_noctrl_lgr.clear()
            rlocus(G, ax=self.ax_noctrl_lgr)
            self.setup_plot_style(self.ax_noctrl_lgr)
            self.ax_noctrl_lgr.set_title('LGR - Sem Controlador', pad=20, 
                                       fontsize=12, color=self.graph_colors['primary'])
            self.ax_noctrl_lgr.set_xlabel('Parte Real', fontsize=10)
            self.ax_noctrl_lgr.set_ylabel('Parte Imaginária', fontsize=10)
            self.canvas_noctrl_lgr.draw()
    
    def update_ctrl_plots(self, G, Gc):
        """Atualiza gráficos com controlador com estilo melhorado e cores"""
        input_type = self.input_var.get()
        sys_ol = G * Gc # Malha aberta com controlador
        sys_cl = feedback(sys_ol, 1) # Malha fechada
        
        if self.show_ctrl_resp_var.get():
            self.ax_ctrl_resp.clear()
            
            if input_type == "Degrau":
                t, y = step_response(sys_cl)
                self.ax_ctrl_resp.plot(t, y, linewidth=2, color=self.graph_colors['primary'])
                self.ax_ctrl_resp.set_title('Resposta ao Degrau - Com Controlador', pad=20, 
                                          fontsize=12, color=self.graph_colors['primary'])
            elif input_type == "Rampa":
                t = np.linspace(0, 10, 1000)
                u = t
                t, y, _ = scipy_lsim(sys_cl, u, t)
                self.ax_ctrl_resp.plot(t, y, linewidth=2, color=self.graph_colors['primary'], label='Saída')
                self.ax_ctrl_resp.plot(t, u, '--', linewidth=2, color=self.graph_colors['secondary'], label='Entrada')
                self.ax_ctrl_resp.set_title('Resposta à Rampa - Com Controlador', pad=20, 
                                          fontsize=12, color=self.graph_colors['primary'])
                self.ax_ctrl_resp.legend()
            elif input_type == "Parábola":
                t = np.linspace(0, 10, 1000)
                u = (t**2) / 2
                t, y, _ = scipy_lsim(sys_cl, u, t)
                self.ax_ctrl_resp.plot(t, y, linewidth=2, color=self.graph_colors['primary'], label='Saída')
                self.ax_ctrl_resp.plot(t, u, '--', linewidth=2, color=self.graph_colors['secondary'], label='Entrada')
                self.ax_ctrl_resp.set_title('Resposta à Parábola - Com Controlador', pad=20, 
                                          fontsize=12, color=self.graph_colors['primary'])
                self.ax_ctrl_resp.legend()
            
            self.setup_plot_style(self.ax_ctrl_resp)
            self.ax_ctrl_resp.set_xlabel('Tempo (s)', fontsize=10)
            self.ax_ctrl_resp.set_ylabel('Saída', fontsize=10)
            self.canvas_ctrl_resp.draw()
        
        if self.show_ctrl_lgr_var.get():
            self.ax_ctrl_lgr.clear()
            rlocus(sys_ol, ax=self.ax_ctrl_lgr) # LGR é da malha aberta
            self.setup_plot_style(self.ax_ctrl_lgr)
            self.ax_ctrl_lgr.set_title('LGR - Com Controlador', pad=20, 
                                     fontsize=12, color=self.graph_colors['primary'])
            self.ax_ctrl_lgr.set_xlabel('Parte Real', fontsize=10)
            self.ax_ctrl_lgr.set_ylabel('Parte Imaginária', fontsize=10)
            self.canvas_ctrl_lgr.draw()
    
    def update_pz_plot(self, G):
        """Atualiza o gráfico de polos e zeros com estilo melhorado e cores"""
        try:
            poles = ctrl_poles(G)
            zeros = ctrl_zeros(G)
            
            self.ax_pz.clear()
            
            if poles.size > 0:
                # Colorir polos de acordo com a estabilidade
                for pole in poles:
                    if pole.real < 0:
                        self.ax_pz.plot(pole.real, pole.imag, 'x', markersize=10, markeredgewidth=2, 
                                      color='#28a745', label='Polo Estável' if pole == poles[0] else "")
                    else:
                        self.ax_pz.plot(pole.real, pole.imag, 'x', markersize=10, markeredgewidth=2, 
                                      color='#dc3545', label='Polo Instável' if pole == poles[0] else "")
            
            if zeros.size > 0:
                self.ax_pz.plot(zeros.real, zeros.imag, 'o', markersize=10, markeredgewidth=2, 
                              fillstyle='none', color='#6f42c1', label='Zeros')
            
            self.setup_plot_style(self.ax_pz)
            self.ax_pz.axhline(0, color='black', linewidth=0.5)
            self.ax_pz.axvline(0, color='black', linewidth=0.5)
            self.ax_pz.set_title('Diagrama de Polos e Zeros do Sistema Original', pad=20, 
                               fontsize=12, color=self.graph_colors['primary'])
            self.ax_pz.set_xlabel('Parte Real', fontsize=10)
            self.ax_pz.set_ylabel('Parte Imaginária', fontsize=10)
            
            if poles.size > 0 or zeros.size > 0:
                self.ax_pz.legend()
            
            self.canvas_pz.draw()
            
            # Atualizar informações textuais
            pole_info = "\n".join([f"Polo {i+1}: {p.real:.4f} {'+' if p.imag >= 0 else ''}{p.imag:.4f}j" 
                                 for i, p in enumerate(poles)])
            zero_info = "\n".join([f"Zero {i+1}: {z.real:.4f} {'+' if z.imag >= 0 else ''}{z.imag:.4f}j" 
                                 for i, z in enumerate(zeros)]) if zeros.size > 0 else "Nenhum zero"
            
            info_text = f"=== Polos do Sistema Original ===\n{pole_info}\n\n=== Zeros do Sistema Original ===\n{zero_info}"
            
            self.pz_info.config(state=tk.NORMAL)
            self.pz_info.delete(1.0, tk.END)
            self.pz_info.insert(tk.END, info_text)
            self.pz_info.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao calcular polos e zeros: {str(e)}")
    
    def show_poles_zeros(self):
        """Mostra os polos e zeros"""
        try:
            num = self.parse_coefficients(self.num_entry.get())
            den = self.parse_coefficients(self.den_entry.get())
            
            if num is None or den is None:
                return
                
            G = tf(num, den)
            self.update_pz_plot(G)
            self.graph_notebook.select(self.pz_frame)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
    
    def generate_pdf_report(self):
        """Gera relatório PDF com melhor formatação usando FPDF2"""
        try:
            # Pergunta ao usuário onde salvar o PDF
            filepath = filedialog.asksaveasfilename(
                title="Salvar Relatório PDF",
                defaultextension=".pdf",
                filetypes=[("Arquivos PDF", "*.pdf"), ("Todos os arquivos", "*.*")]
            )
            
            if not filepath:
                return  # Usuário cancelou

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Configurações de fonte
            pdf.set_font("helvetica", 'B', 16)
            pdf.cell(0, 10, "Relatório de Análise de Sistema de Controle", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            pdf.ln(10)
            
            # Informações do sistema
            pdf.set_font("helvetica", 'B', 12)
            pdf.cell(0, 10, "Configuração do Sistema:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", '', 10)
            
            num = self.parse_coefficients(self.num_entry.get())
            den = self.parse_coefficients(self.den_entry.get())
            
            if num is None or den is None:
                return
                
            num_str = " ".join([str(x) for x in num])
            den_str = " ".join([str(x) for x in den])
            
            pdf.cell(0, 6, f"Numerador: {num_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 6, f"Denominador: {den_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 6, f"Tipo de entrada: {self.input_var.get()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 6, f"Tipo de controlador: {self.ctrl_type.get()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(10)
            
            # Salvar gráficos temporariamente
            temp_files = []
            
            def add_plot_to_pdf(fig, title):
                pdf.set_font("helvetica", 'B', 12)
                pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                temp_file = self.save_temp_plot(fig)
                temp_files.append(temp_file)
                if pdf.get_y() + 80 > 280:
                    pdf.add_page()
                pdf.image(temp_file.name, x=10, w=190)
                pdf.ln(5)

            # Adicionar gráficos ao PDF
            if self.show_noctrl_resp_var.get():
                add_plot_to_pdf(self.fig_noctrl_resp, "Resposta Temporal - Sem Controlador")
            
            if self.show_ctrl_resp_var.get():
                add_plot_to_pdf(self.fig_ctrl_resp, "Resposta Temporal - Com Controlador")
            
            if self.show_noctrl_lgr_var.get():
                add_plot_to_pdf(self.fig_noctrl_lgr, "LGR - Sem Controlador")
            
            if self.show_ctrl_lgr_var.get():
                add_plot_to_pdf(self.fig_ctrl_lgr, "LGR - Com Controlador")
            
            # Adicionar polos e zeros em uma nova página
            pdf.add_page()
            add_plot_to_pdf(self.fig_pz, "Diagrama de Polos e Zeros do Sistema Original")
            
            # Informações textuais de Polos e Zeros
            G = tf(num, den)
            poles = ctrl_poles(G)
            zeros = ctrl_zeros(G)
            
            pdf.set_font("helvetica", 'B', 11)
            pdf.cell(0, 8, "Polos do Sistema Original:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", '', 10)
            for i, p in enumerate(poles):
                pdf.cell(0, 6, f"  Polo {i+1}: {p.real:.4f} {'+' if p.imag >= 0 else ''}{p.imag:.4f}j", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.ln(3)
            pdf.set_font("helvetica", 'B', 11)
            pdf.cell(0, 8, "Zeros do Sistema Original:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", '', 10)
            
            if zeros.size > 0:
                for i, z in enumerate(zeros):
                    pdf.cell(0, 6, f"  Zero {i+1}: {z.real:.4f} {'+' if z.imag >= 0 else ''}{z.imag:.4f}j", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 6, "  Nenhum zero.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            # Adicionar análise de estabilidade
            pdf.add_page()
            pdf.set_font("helvetica", 'B', 12)
            pdf.cell(0, 10, "Análise de Estabilidade:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            s_poly = self.create_polynomial(den)
            rh_array, stability = self.routh_hurwitz(s_poly)
            
            pdf.set_font("helvetica", '', 10)
            pdf.cell(0, 6, f"Resultado: {stability}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            # Salvar o PDF
            pdf.output(filepath)
            self.status_label.config(text=f"Relatório salvo em: {filepath}")
            
            # Limpar arquivos temporários
            for temp_file in temp_files:
                temp_file.close()
                os.unlink(temp_file.name)
            
        except PermissionError:
            messagebox.showerror("Erro de Permissão", "Permissão negada para salvar o arquivo.\nVerifique se o arquivo não está aberto em outro programa ou escolha outro local.")
        except Exception as e:
            messagebox.showerror("Erro ao Gerar PDF", f"Falha ao gerar relatório PDF: {str(e)}")
            self.status_label.config(text="Erro ao gerar relatório")
    
    def save_temp_plot(self, fig):
        """Salva figura temporariamente para o PDF com alta qualidade"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(temp_file.name, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        return temp_file
    
    def clear_all(self):
        """Limpa todos os gráficos e campos"""
        # Limpar gráficos
        for ax in [self.ax_noctrl_resp, self.ax_ctrl_resp, 
                  self.ax_noctrl_lgr, self.ax_ctrl_lgr, self.ax_pz]:
            ax.clear()
            self.setup_plot_style(ax)
        
        for canvas in [self.canvas_noctrl_resp, self.canvas_ctrl_resp,
                      self.canvas_noctrl_lgr, self.canvas_ctrl_lgr, self.canvas_pz]:
            canvas.draw()
        
        # Limpar informações textuais
        self.pz_info.config(state=tk.NORMAL)
        self.pz_info.delete(1.0, tk.END)
        self.pz_info.insert(tk.END, "Informações sobre polos e zeros serão exibidas aqui...")
        self.pz_info.config(state=tk.DISABLED)
        
        # Resetar campos de entrada
        self.num_entry.delete(0, tk.END)
        self.num_entry.insert(0, "1")
        self.den_entry.delete(0, tk.END)
        self.den_entry.insert(0, "1 1")
        
        # Resetar outros campos
        self.input_var.set("Degrau")
        self.ctrl_type.set("P")
        self.update_controller_fields()
        
        self.status_label.config(text="Tudo limpo. Pronto para nova análise.")
    
    def setup_shortcuts(self):
        """Configura atalhos de teclado"""
        self.root.bind('<Control-g>', lambda e: self.generate_plots())
        self.root.bind('<Control-p>', lambda e: self.generate_pdf_report())
        self.root.bind('<Control-l>', lambda e: self.clear_all())
        self.root.bind('<Control-z>', lambda e: self.show_poles_zeros())
        self.root.bind('<Control-n>', lambda e: self.clear_all())
        self.root.bind('<F1>', lambda e: self.show_help())
    
    def run(self):
        """Executa a aplicação"""
        self.root.mainloop()

def main():
    try:
        root = tk.Tk()
        app = AdvancedControlSystemApp(root)
        app.run()
    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        try:
            messagebox.showerror("Erro Fatal", f"Ocorreu um erro crítico e a aplicação precisa fechar:\n\n{str(e)}")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()