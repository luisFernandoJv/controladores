import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from control import tf, step_response, feedback, rlocus
from scipy.signal import lsim as scipy_lsim
import tempfile
from control import poles as ctrl_poles, zeros as ctrl_zeros
import os
import sys
from matplotlib.ticker import AutoMinorLocator

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
        self.setup_window()
        self.setup_variables()
        self.setup_styles()
        self.create_widgets()
        self.setup_shortcuts()
        self.setup_responsive_layout()
        
    def setup_window(self):
        """Configura a janela principal"""
        self.root.title("Sistema Avançado de Análise de Controladores")
        self.root.geometry("1400x900")
        self.root.minsize(800, 600)  # Tamanho mínimo reduzido para telas menores
        self.root.configure(bg='#f0f0f0')
        
        # Configurações para alta DPI (melhor visualização em telas 4K)
        self.root.tk.call('tk', 'scaling', 1.5 if self.root.winfo_fpixels('1i') > 100 else 1.0)
    
    def setup_responsive_layout(self):
        """Configura o layout responsivo"""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Ajusta o tamanho da fonte baseado no tamanho da tela
        screen_width = self.root.winfo_screenwidth()
        base_font_size = 9 if screen_width < 1920 else 11
        
        style = ttk.Style()
        style.configure('.', font=('Segoe UI', base_font_size))
        style.configure('Header.TLabel', font=('Segoe UI', base_font_size+2, 'bold'))
        style.configure('Title.TLabel', font=('Segoe UI', base_font_size+5, 'bold'))
    
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
        """Configura os estilos visuais"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurações gerais
        style.configure('.', font=('Segoe UI', 9))
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TLabel', background='#f5f5f5')
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=5)
        
        # Estilos personalizados
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Graph.TFrame', background='white', relief=tk.SUNKEN, borderwidth=1)
        style.configure('Red.TButton', foreground='red')
        style.configure('Large.TButton', font=('Segoe UI', 10, 'bold'), padding=8)
    
    def create_widgets(self):
        """Cria todos os widgets da interface"""
        self.create_main_frames()
        self.create_control_panel()
        self.create_graph_area()
        self.create_status_bar()
    
    def create_main_frames(self):
        """Cria os frames principais com layout responsivo"""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controle (25% da largura)
        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Frame de gráficos (75% da largura)
        self.graph_area = ttk.Frame(self.main_frame)
        self.graph_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def create_control_panel(self):
        """Cria o painel de controle esquerdo"""
        self.create_system_frame()
        self.create_input_frame()
        self.create_controller_frame()
        self.create_graph_selection_frame()
        self.create_action_buttons()
    
    def create_system_frame(self):
        """Frame de configuração do sistema"""
        sys_frame = ttk.LabelFrame(self.control_panel, text="Configuração do Sistema", padding=10)
        sys_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sys_frame, text="Numerador:").grid(row=0, column=0, sticky="e", pady=2)
        self.num_entry = ttk.Entry(sys_frame)
        self.num_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.num_entry.insert(0, "1")
        self.num_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ex: 1 2 3 para [1, 2, 3]"))
        
        ttk.Label(sys_frame, text="Denominador:").grid(row=1, column=0, sticky="e", pady=2)
        self.den_entry = ttk.Entry(sys_frame)
        self.den_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.den_entry.insert(0, "1 1")
        self.den_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ex: 1 5 6 para [1, 5, 6]"))
    
    def show_tooltip(self, message):
        """Mostra uma dica contextual"""
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{self.root.winfo_pointerx()+10}+{self.root.winfo_pointery()+10}")
        label = ttk.Label(tooltip, text=message, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()
        self.root.after(3000, tooltip.destroy)
    
    def create_input_frame(self):
        """Frame de seleção de tipo de entrada"""
        input_frame = ttk.LabelFrame(self.control_panel, text="Tipo de Entrada", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_var = tk.StringVar(value="Degrau")
        inputs = ["Degrau", "Rampa", "Parábola"]
        for i, inp in enumerate(inputs):
            rb = ttk.Radiobutton(input_frame, text=inp, variable=self.input_var, value=inp)
            rb.grid(row=0, column=i, padx=5, pady=2, sticky="w")
    
    def create_controller_frame(self):
        """Frame de configuração do controlador"""
        ctrl_frame = ttk.LabelFrame(self.control_panel, text="Configuração do Controlador", padding=10)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ctrl_frame, text="Tipo:").grid(row=0, column=0, sticky="e", pady=2)
        self.ctrl_type = ttk.Combobox(ctrl_frame, values=["P", "PI", "PD", "PID", "Avanço", "Atraso", "Avanço-Atraso"])
        self.ctrl_type.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.ctrl_type.set("P")
        self.ctrl_type.bind("<<ComboboxSelected>>", self.update_controller_fields)
        
        self.ctrl_params_frame = ttk.Frame(ctrl_frame)
        self.ctrl_params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.update_controller_fields()
    
    def create_graph_selection_frame(self):
        """Frame de seleção de gráficos"""
        graph_select_frame = ttk.LabelFrame(self.control_panel, text="Gráficos a Exibir", padding=10)
        graph_select_frame.pack(fill=tk.X, pady=5)
        
        self.show_noctrl_resp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(graph_select_frame, text="Resposta sem controlador", 
                       variable=self.show_noctrl_resp_var).pack(anchor="w", pady=2)
        
        self.show_ctrl_resp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(graph_select_frame, text="Resposta com controlador", 
                       variable=self.show_ctrl_resp_var).pack(anchor="w", pady=2)
        
        self.show_noctrl_lgr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(graph_select_frame, text="LGR sem controlador", 
                       variable=self.show_noctrl_lgr_var).pack(anchor="w", pady=2)
        
        self.show_ctrl_lgr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(graph_select_frame, text="LGR com controlador", 
                       variable=self.show_ctrl_lgr_var).pack(anchor="w", pady=2)
    
    def create_action_buttons(self):
        """Cria os botões de ação"""
        btn_frame = ttk.Frame(self.control_panel)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Gerar Gráficos", command=self.generate_plots, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="Polos/Zeros", command=self.show_poles_zeros,
                  style='Large.TButton').pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        btn_frame2 = ttk.Frame(self.control_panel)
        btn_frame2.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame2, text="Gerar PDF", command=self.generate_pdf_report,
                  style='Large.TButton').pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame2, text="Limpar", command=self.clear_all, 
                  style='Large.TButton').pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
    
    def create_graph_area(self):
        """Cria a área de exibição dos gráficos com layout responsivo"""
        self.graph_notebook = ttk.Notebook(self.graph_area)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Abas
        self.create_response_tab()
        self.create_lgr_tab()
        self.create_pz_tab()
    
    def create_response_tab(self):
        """Aba de resposta temporal com layout melhorado"""
        self.resp_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.resp_frame, text="Resposta Temporal")
        
        # Frame contêiner para os dois gráficos
        resp_container = ttk.Frame(self.resp_frame)
        resp_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Gráfico sem controlador
        self.noctrl_resp_frame = ttk.LabelFrame(resp_container, text="Sem Controlador", padding=5)
        self.noctrl_resp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_noctrl_resp = plt.Figure(figsize=(6, 4), dpi=100, facecolor='#f5f5f5')
        self.ax_noctrl_resp = self.fig_noctrl_resp.add_subplot(111)
        self.setup_plot_style(self.ax_noctrl_resp)
        self.canvas_noctrl_resp = FigureCanvasTkAgg(self.fig_noctrl_resp, master=self.noctrl_resp_frame)
        self.canvas_noctrl_resp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico com controlador
        self.ctrl_resp_frame = ttk.LabelFrame(resp_container, text="Com Controlador", padding=5)
        self.ctrl_resp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_ctrl_resp = plt.Figure(figsize=(6, 4), dpi=100, facecolor='#f5f5f5')
        self.ax_ctrl_resp = self.fig_ctrl_resp.add_subplot(111)
        self.setup_plot_style(self.ax_ctrl_resp)
        self.canvas_ctrl_resp = FigureCanvasTkAgg(self.fig_ctrl_resp, master=self.ctrl_resp_frame)
        self.canvas_ctrl_resp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_lgr_tab(self):
        """Aba de LGR com layout melhorado"""
        self.lgr_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.lgr_frame, text="Lugar Geométrico das Raízes")
        
        # Frame contêiner para os dois gráficos
        lgr_container = ttk.Frame(self.lgr_frame)
        lgr_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Gráfico sem controlador
        self.noctrl_lgr_frame = ttk.LabelFrame(lgr_container, text="Sem Controlador", padding=5)
        self.noctrl_lgr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_noctrl_lgr = plt.Figure(figsize=(6, 4), dpi=100, facecolor='#f5f5f5')
        self.ax_noctrl_lgr = self.fig_noctrl_lgr.add_subplot(111)
        self.setup_plot_style(self.ax_noctrl_lgr)
        self.canvas_noctrl_lgr = FigureCanvasTkAgg(self.fig_noctrl_lgr, master=self.noctrl_lgr_frame)
        self.canvas_noctrl_lgr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico com controlador
        self.ctrl_lgr_frame = ttk.LabelFrame(lgr_container, text="Com Controlador", padding=5)
        self.ctrl_lgr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_ctrl_lgr = plt.Figure(figsize=(6, 4), dpi=100, facecolor='#f5f5f5')
        self.ax_ctrl_lgr = self.fig_ctrl_lgr.add_subplot(111)
        self.setup_plot_style(self.ax_ctrl_lgr)
        self.canvas_ctrl_lgr = FigureCanvasTkAgg(self.fig_ctrl_lgr, master=self.ctrl_lgr_frame)
        self.canvas_ctrl_lgr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_plot_style(self, ax):
        """Configura o estilo dos gráficos para melhor visualização"""
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_facecolor('#f9f9f9')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=3)
    
    def create_pz_tab(self):
        """Aba de polos e zeros com layout melhorado"""
        self.pz_frame = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.pz_frame, text="Polos e Zeros")
        
        # Frame principal
        pz_container = ttk.Frame(self.pz_frame)
        pz_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gráfico
        graph_frame = ttk.LabelFrame(pz_container, text="Diagrama de Polos e Zeros", padding=5)
        graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.fig_pz = plt.Figure(figsize=(8, 6), dpi=100, facecolor='#f5f5f5')
        self.ax_pz = self.fig_pz.add_subplot(111)
        self.setup_plot_style(self.ax_pz)
        self.canvas_pz = FigureCanvasTkAgg(self.fig_pz, master=graph_frame)
        self.canvas_pz.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Área de informações textuais
        info_frame = ttk.LabelFrame(pz_container, text="Informações", padding=5)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.pz_info = tk.Text(info_frame, wrap=tk.WORD, height=10, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.pz_info.yview)
        self.pz_info.configure(yscrollcommand=scrollbar.set)
        
        self.pz_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pz_info.insert(tk.END, "Informações sobre polos e zeros serão exibidas aqui...")
        self.pz_info.config(state=tk.DISABLED)
    
    def create_status_bar(self):
        """Cria a barra de status com informações úteis"""
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Pronto. Use Ctrl+G para gerar gráficos.")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Adiciona um botão de ajuda
        help_btn = ttk.Button(self.status_bar, text="Ajuda", command=self.show_help,
                             width=8, style='Small.TButton')
        help_btn.pack(side=tk.RIGHT, padx=5)
    
    def show_help(self):
        """Mostra uma janela de ajuda"""
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
Ctrl+L - Limpar tudo"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Ajuda")
        help_window.geometry("500x400")
        
        text = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED, font=('Segoe UI', 10))
        text.pack(fill=tk.BOTH, expand=True)
        
        close_btn = ttk.Button(help_window, text="Fechar", command=help_window.destroy)
        close_btn.pack(pady=10)
    
    def update_controller_fields(self, event=None):
        """Atualiza os campos do controlador"""
        for widget in self.ctrl_params_frame.winfo_children():
            widget.destroy()
        
        ctrl_type = self.ctrl_type.get()
        row = 0
        
        # Kp (comum a todos)
        ttk.Label(self.ctrl_params_frame, text="Kp:").grid(row=row, column=0, sticky="e", pady=2)
        self.kp_entry = ttk.Entry(self.ctrl_params_frame, width=10)
        self.kp_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        self.kp_entry.insert(0, "1.0")
        self.kp_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ganho proporcional"))
        row += 1
        
        # Campos específicos
        if ctrl_type in ["PI", "PID"]:
            ttk.Label(self.ctrl_params_frame, text="Ki:").grid(row=row, column=0, sticky="e", pady=2)
            self.ki_entry = ttk.Entry(self.ctrl_params_frame, width=10)
            self.ki_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            self.ki_entry.insert(0, "0.1")
            self.ki_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ganho integral"))
            row += 1
        
        if ctrl_type in ["PD", "PID"]:
            ttk.Label(self.ctrl_params_frame, text="Kd:").grid(row=row, column=0, sticky="e", pady=2)
            self.kd_entry = ttk.Entry(self.ctrl_params_frame, width=10)
            self.kd_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            self.kd_entry.insert(0, "0.1")
            self.kd_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Ganho derivativo"))
            row += 1
        
        if ctrl_type in ["Avanço", "Atraso", "Avanço-Atraso"]:
            ttk.Label(self.ctrl_params_frame, text="Numerador:").grid(row=row, column=0, sticky="e", pady=2)
            self.num_c_entry = ttk.Entry(self.ctrl_params_frame)
            self.num_c_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            self.num_c_entry.insert(0, "1")
            self.num_c_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do numerador"))
            row += 1
            
            ttk.Label(self.ctrl_params_frame, text="Denominador:").grid(row=row, column=0, sticky="e", pady=2)
            self.den_c_entry = ttk.Entry(self.ctrl_params_frame)
            self.den_c_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            self.den_c_entry.insert(0, "1 1")
            self.den_c_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do denominador"))
            row += 1
            
            if ctrl_type == "Avanço-Atraso":
                ttk.Label(self.ctrl_params_frame, text="Numerador Atraso:").grid(row=row, column=0, sticky="e", pady=2)
                self.num_at_entry = ttk.Entry(self.ctrl_params_frame)
                self.num_at_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
                self.num_at_entry.insert(0, "1")
                self.num_at_entry.bind("<FocusIn>", lambda e: self.show_tooltip("Coeficientes do numerador (atraso)"))
                row += 1
                
                ttk.Label(self.ctrl_params_frame, text="Denominador Atraso:").grid(row=row, column=0, sticky="e", pady=2)
                self.den_at_entry = ttk.Entry(self.ctrl_params_frame)
                self.den_at_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
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
                # ### CORREÇÃO APLICADA AQUI ###
                # O ganho Kp agora é multiplicado ao resultado final.
                return kp * G_avanco * G_atraso
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Valor inválido: {str(e)}")
            return None
    
    def generate_plots(self):
        """Gera todos os gráficos selecionados"""
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
        """Atualiza gráficos sem controlador com estilo melhorado"""
        input_type = self.input_var.get()
        
        if self.show_noctrl_resp_var.get():
            self.ax_noctrl_resp.clear()
            
            if input_type == "Degrau":
                t, y = step_response(feedback(G, 1))
                self.ax_noctrl_resp.plot(t, y, linewidth=2, color='blue')
                self.ax_noctrl_resp.set_title('Resposta ao Degrau - Sem Controlador', pad=20)
            elif input_type == "Rampa":
                t = np.linspace(0, 10, 1000)
                u = t
                sys_fb = feedback(G, 1)
                t, y, _ = scipy_lsim(sys_fb, u, t)
                self.ax_noctrl_resp.plot(t, y, linewidth=2, color='green', label='Saída')
                self.ax_noctrl_resp.plot(t, u, 'r--', label='Entrada')
                self.ax_noctrl_resp.set_title('Resposta à Rampa - Sem Controlador', pad=20)
                self.ax_noctrl_resp.legend()
            elif input_type == "Parábola":
                t = np.linspace(0, 10, 1000)
                u = (t**2) / 2
                sys_fb = feedback(G, 1)
                t, y, _ = scipy_lsim(sys_fb, u, t)
                self.ax_noctrl_resp.plot(t, y, linewidth=2, color='red', label='Saída')
                self.ax_noctrl_resp.plot(t, u, 'b--', label='Entrada')
                self.ax_noctrl_resp.set_title('Resposta à Parábola - Sem Controlador', pad=20)
                self.ax_noctrl_resp.legend()
            
            self.setup_plot_style(self.ax_noctrl_resp)
            self.ax_noctrl_resp.set_xlabel('Tempo (s)', fontsize=10)
            self.ax_noctrl_resp.set_ylabel('Saída', fontsize=10)
            self.canvas_noctrl_resp.draw()
        
        if self.show_noctrl_lgr_var.get():
            self.ax_noctrl_lgr.clear()
            rlocus(G, ax=self.ax_noctrl_lgr)
            self.setup_plot_style(self.ax_noctrl_lgr)
            self.ax_noctrl_lgr.set_title('LGR - Sem Controlador', pad=20)
            self.ax_noctrl_lgr.set_xlabel('Parte Real', fontsize=10)
            self.ax_noctrl_lgr.set_ylabel('Parte Imaginária', fontsize=10)
            self.canvas_noctrl_lgr.draw()
    
    def update_ctrl_plots(self, G, Gc):
        """Atualiza gráficos com controlador com estilo melhorado"""
        input_type = self.input_var.get()
        sys_ol = G * Gc # Malha aberta com controlador
        sys_cl = feedback(sys_ol, 1) # Malha fechada
        
        if self.show_ctrl_resp_var.get():
            self.ax_ctrl_resp.clear()
            
            if input_type == "Degrau":
                t, y = step_response(sys_cl)
                self.ax_ctrl_resp.plot(t, y, linewidth=2, color='blue')
                self.ax_ctrl_resp.set_title('Resposta ao Degrau - Com Controlador', pad=20)
            elif input_type == "Rampa":
                t = np.linspace(0, 10, 1000)
                u = t
                t, y, _ = scipy_lsim(sys_cl, u, t)
                self.ax_ctrl_resp.plot(t, y, linewidth=2, color='green', label='Saída')
                self.ax_ctrl_resp.plot(t, u, 'r--', label='Entrada')
                self.ax_ctrl_resp.set_title('Resposta à Rampa - Com Controlador', pad=20)
                self.ax_ctrl_resp.legend()
            elif input_type == "Parábola":
                t = np.linspace(0, 10, 1000)
                u = (t**2) / 2
                t, y, _ = scipy_lsim(sys_cl, u, t)
                self.ax_ctrl_resp.plot(t, y, linewidth=2, color='red', label='Saída')
                self.ax_ctrl_resp.plot(t, u, 'b--', label='Entrada')
                self.ax_ctrl_resp.set_title('Resposta à Parábola - Com Controlador', pad=20)
                self.ax_ctrl_resp.legend()
            
            self.setup_plot_style(self.ax_ctrl_resp)
            self.ax_ctrl_resp.set_xlabel('Tempo (s)', fontsize=10)
            self.ax_ctrl_resp.set_ylabel('Saída', fontsize=10)
            self.canvas_ctrl_resp.draw()
        
        if self.show_ctrl_lgr_var.get():
            self.ax_ctrl_lgr.clear()
            rlocus(sys_ol, ax=self.ax_ctrl_lgr) # LGR é da malha aberta
            self.setup_plot_style(self.ax_ctrl_lgr)
            self.ax_ctrl_lgr.set_title('LGR - Com Controlador', pad=20)
            self.ax_ctrl_lgr.set_xlabel('Parte Real', fontsize=10)
            self.ax_ctrl_lgr.set_ylabel('Parte Imaginária', fontsize=10)
            self.canvas_ctrl_lgr.draw()
    
    def update_pz_plot(self, G):
        """Atualiza o gráfico de polos e zeros com estilo melhorado"""
        try:
            poles = ctrl_poles(G)
            zeros = ctrl_zeros(G)
            
            self.ax_pz.clear()
            
            if poles.size > 0:
                self.ax_pz.plot(poles.real, poles.imag, 'x', markersize=10, markeredgewidth=2, label='Polos')
            if zeros.size > 0:
                self.ax_pz.plot(zeros.real, zeros.imag, 'o', markersize=10, markeredgewidth=2, fillstyle='none', label='Zeros')
            
            self.setup_plot_style(self.ax_pz)
            self.ax_pz.axhline(0, color='black', linewidth=0.5)
            self.ax_pz.axvline(0, color='black', linewidth=0.5)
            self.ax_pz.set_title('Diagrama de Polos e Zeros do Sistema Original', pad=20)
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

            # ### CORREÇÃO APLICADA AQUI ###
            # As importações foram movidas para o topo do arquivo com tratamento de erro.
            # O código aqui agora usa FPDF, XPos, e YPos que já foram importados corretamente.
            
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Configurações de fonte - usando fontes padrão do FPDF2
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
                # Adiciona uma nova página se não houver espaço suficiente
                if pdf.get_y() + 80 > 280: # 80mm é uma estimativa da altura da imagem
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
        self.root.bind('<Control-r>', lambda e: self.generate_pdf_report())
        self.root.bind('<Control-l>', lambda e: self.clear_all())
        self.root.bind('<Control-p>', lambda e: self.show_poles_zeros())
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
        # Em vez de sys.exit, podemos mostrar o erro em uma messagebox se o tkinter ainda estiver de pé
        try:
            messagebox.showerror("Erro Fatal", f"Ocorreu um erro crítico e a aplicação precisa fechar:\n\n{str(e)}")
        except:
            pass # Se nem o tkinter funcionar, o print no console é o fallback
        sys.exit(1)

if __name__ == "__main__":
    main()