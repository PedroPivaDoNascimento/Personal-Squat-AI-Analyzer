### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/settings.py
*Saved at: 23/06/2026, 15:16:44*

**[REMOVED]**
```
(from line ~40)
    'squat_analyzer',

```
**[ADDED]**
```
40        'squat_analysis_app.squat_analyzer',
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/manage.py
*Saved at: 23/06/2026, 15:13:13*

**[REMOVED]**
```
(from line ~6)


```
**[ADDED]**
```
8         # O ponto crucial aqui: define o diretório do projeto como raiz
```
**[ADDED]**
```
10        
```
**[ADDED]**
```
19        
20        # Esta linha garante que o diretório atual (onde está o manage.py) 
21        # esteja no sys.path, permitindo importar 'squat_analyzer'
```
**[REMOVED]**
```
(from line ~24)


```
**[REMOVED]**
```
(from line ~25)
    main()

```
**[ADDED]**
```
25        main()
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/squat_analyzer/templates/squat_analyzer/sagittal_analysis.html
*Saved at: 23/06/2026, 14:56:39*

**[ADDED]**
```
1     {% extends 'squat_analyzer/base.html' %}
2     
3     {% block title %}{{ title }}{% endblock %}
4     
5     {% block content %}
6     <a href="{% url 'index' %}" class="btn btn-secondary">← Voltar</a>
7     
8     <h1>{{ title }}</h1>
9     
10    {% if result %}
11        <!-- EXIBIÇÃO DE RESULTADOS -->
12        <div class="alert alert-success">
13            <h2>✅ Análise Concluída!</h2>
14            <p><strong>Nome:</strong> {{ person_name }}</p>
15            <p><strong>Repetições Detectadas:</strong> {{ result.repetitions_detected }}</p>
16        </div>
17    
18        <h2>Resumo por Repetição</h2>
19        {% for detail in result.repetition_details %}
20        <div class="card">
21            <h3>Repetição {{ detail.repetition }} ({{ detail.timestamp|floatformat:2 }}s)</h3>
22            <table>
23                <tr><th>Métrica</th><th>Status</th><th>Contagem</th></tr>
24                <tr><td>Tronco</td><td class="{{ detail.trunk_status|yesno:'error,ok' }}">{{ detail.trunk_status }}</td><td>{{ detail.trunk_count }}</td></tr>
25                <tr><td>Joelho</td><td class="{{ detail.knee_status|yesno:'error,ok' }}">{{ detail.knee_status }}</td><td>{{ detail.knee_count }}</td></tr>
26                <tr><td>Cabeça</td><td class="{{ detail.head_status|yesno:'error,ok' }}">{{ detail.head_status }}</td><td>{{ detail.head_count }}</td></tr>
27                <tr><td>Calcanhar</td><td class="{{ detail.heel_status|yesno:'error,ok' }}">{{ detail.heel_status }}</td><td>{{ detail.heel_count }}</td></tr>
28            </table>
29            <div class="feedback">
30                <strong>Feedback:</strong>
31                <ul>
32                    {% for msg in detail.feedback %}
33                    <li>{{ msg }}</li>
34                    {% endfor %}
35                </ul>
36            </div>
37        </div>
38        {% empty %}
39        <div class="alert alert-info">Nenhuma repetição detectada com os parâmetros atuais.</div>
40        {% endfor %}
41    
42        <h2>Dados Detalhados</h2>
43        {% if result.dataframes %}
44            {% for df_name, df_data in result.dataframes.items %}
45            <div class="card">
46                <h3>{{ df_name }}</h3>
47                {% if df_data %}
48                <table>
49                    <thead>
50                        <tr>
51                            {% for key in df_data.0.keys %}
52                            <th>{{ key }}</th>
53                            {% endfor %}
54                        </tr>
55                    </thead>
56                    <tbody>
57                        {% for row in df_data %}
58                        <tr>
59                            {% for value in row.values %}
60                            <td>{{ value }}</td>
61                            {% endfor %}
62                        </tr>
63                        {% endfor %}
64                    </tbody>
65                </table>
66                {% else %}
67                <p class="alert alert-info">Nenhum dado registrado.</p>
68                {% endif %}
69            </div>
70            {% endfor %}
71        {% endif %}
72    
73        <a href="{% url 'sagittal_analysis' side %}" class="btn">Nova Análise</a>
74    
75    {% else %}
76        <!-- FORMULÁRIO DE UPLOAD -->
77        <form method="post" enctype="multipart/form-data">
78            {% csrf_token %}
79            
80            <label for="person_name">Nome da Pessoa:</label>
81            <input type="text" id="person_name" name="person_name" required>
82            
83            <label for="user_height_cm">Altura (cm):</label>
84            <input type="number" id="user_height_cm" name="user_height_cm" 
85                   min="100" max="250" value="170" required>
86            
87            <label for="video">Enviar Vídeo ({{ analysis_type|upper }} - {{ side|capfirst }}):</label>
88            <input type="file" id="video" name="video" accept=".mp4,.avi,.mov" required>
89            
90            <h3>Parâmetros de Avaliação</h3>
91            <div class="form-row">
92                <div class="form-col">
93                    <label for="descent_threshold">Sensibilidade da Descida:</label>
94                    <input type="range" id="descent_threshold" name="descent_threshold" 
95                           min="0.01" max="0.10" step="0.005" value="0.05">
96                    <span id="descent_value">0.05</span>
97                    <p class="help-text">Percentual de movimento da orelha para baixo.</p>
98                    
99                    <label for="trunk_error_threshold">Tolerância Tronco:</label>
100                   <input type="range" id="trunk_error_threshold" name="trunk_error_threshold" 
101                          min="1" max="150" step="1" value="23">
102                   <span id="trunk_value">23</span>
103                   <p class="help-text">Instantes permitidos de desvio do tronco.</p>
104                   
105                   <label for="head_error_threshold">Tolerância Cabeça:</label>
106                   <input type="range" id="head_error_threshold" name="head_error_threshold" 
107                          min="1" max="150" step="1" value="2">
108                   <span id="head_value">2</span>
109                   <p class="help-text">Instantes permitidos de desvio da cabeça.</p>
110               </div>
111               
112               <div class="form-col">
113                   <label for="ascent_return_threshold">Retorno na Subida:</label>
114                   <input type="range" id="ascent_return_threshold" name="ascent_return_threshold" 
115                          min="0.005" max="0.05" step="0.005" value="0.02">
116                   <span id="ascent_value">0.02</span>
117                   <p class="help-text">Proximidade da posição inicial.</p>
118                   
119                   <label for="knee_error_threshold">Tolerância Joelho:</label>
120                   <input type="range" id="knee_error_threshold" name="knee_error_threshold" 
121                          min="1" max="150" step="1" value="6">
122                   <span id="knee_value">6</span>
123                   <p class="help-text">Instantes permitidos de desvio do joelho.</p>
124                   
125                   <label for="foot_error_threshold">Tolerância Calcanhar:</label>
126                   <input type="range" id="foot_error_threshold" name="foot_error_threshold" 
127                          min="1" max="150" step="1" value="8">
128                   <span id="foot_value">8</span>
129                   <p class="help-text">Instantes permitidos de calcanhar levantado.</p>
130               </div>
131           </div>
132           
133           <button type="submit" class="btn">Analisar Vídeo</button>
134       </form>
135   
136       <script>
137           document.querySelectorAll('input[type="range"]').forEach(slider => {
138               slider.addEventListener('input', function() {
139                   const span = document.getElementById(this.id.replace('_threshold', '_value'));
140                   if (span) span.textContent = this.value;
141               });
142           });
143       </script>
144   {% endif %}
145   {% endblock %}
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/squat_analyzer/templates/squat_analyzer/frontal_analysis.html
*Saved at: 23/06/2026, 14:55:58*

**[ADDED]**
```
1     {% extends 'squat_analyzer/base.html' %}
2     
3     {% block title %}{{ title }}{% endblock %}
4     
5     {% block content %}
6     <a href="{% url 'index' %}" class="btn btn-secondary">← Voltar</a>
7     
8     <h1>{{ title }}</h1>
9     
10    {% if result %}
11        <!-- EXIBIÇÃO DE RESULTADOS -->
12        <div class="alert alert-success">
13            <h2>✅ Análise Concluída!</h2>
14            <p><strong>Nome:</strong> {{ person_name }}</p>
15            <p><strong>Repetições Detectadas:</strong> {{ result.repetitions_detected }}</p>
16        </div>
17    
18        <h2>Resumo por Repetição</h2>
19        {% for detail in result.repetition_details %}
20        <div class="card">
21            <h3>Repetição {{ detail.repetition }} ({{ detail.timestamp|floatformat:2 }}s)</h3>
22            <table>
23                <tr><th>Métrica</th><th>Status</th><th>Contagem</th></tr>
24                <tr><td>Quadril</td><td class="{{ detail.hip_status|yesno:'error,ok' }}">{{ detail.hip_status }}</td><td>{{ detail.hip_count }}</td></tr>
25                <tr><td>Joelho</td><td class="{{ detail.knee_status|yesno:'error,ok' }}">{{ detail.knee_status }}</td><td>{{ detail.knee_count }}</td></tr>
26                <tr><td>Pé</td><td class="{{ detail.foot_status|yesno:'error,ok' }}">{{ detail.foot_status }}</td><td>{{ detail.foot_count }}</td></tr>
27            </table>
28            <div class="feedback">
29                <strong>Feedback:</strong>
30                <ul>
31                    {% for msg in detail.feedback %}
32                    <li>{{ msg }}</li>
33                    {% endfor %}
34                </ul>
35            </div>
36        </div>
37        {% empty %}
38        <div class="alert alert-info">Nenhuma repetição detectada com os parâmetros atuais.</div>
39        {% endfor %}
40    
41        <h2>Dados Detalhados</h2>
42        {% if result.dataframes %}
43            {% for df_name, df_data in result.dataframes.items %}
44            <div class="card">
45                <h3>{{ df_name }}</h3>
46                {% if df_data %}
47                <table>
48                    <thead>
49                        <tr>
50                            {% for key in df_data.0.keys %}
51                            <th>{{ key }}</th>
52                            {% endfor %}
53                        </tr>
54                    </thead>
55                    <tbody>
56                        {% for row in df_data %}
57                        <tr>
58                            {% for value in row.values %}
59                            <td>{{ value }}</td>
60                            {% endfor %}
61                        </tr>
62                        {% endfor %}
63                    </tbody>
64                </table>
65                {% else %}
66                <p class="alert alert-info">Nenhum dado registrado.</p>
67                {% endif %}
68            </div>
69            {% endfor %}
70        {% endif %}
71    
72        <a href="{% url 'frontal_analysis' side %}" class="btn">Nova Análise</a>
73    
74    {% else %}
75        <!-- FORMULÁRIO DE UPLOAD -->
76        <form method="post" enctype="multipart/form-data">
77            {% csrf_token %}
78            
79            <label for="person_name">Nome da Pessoa:</label>
80            <input type="text" id="person_name" name="person_name" required>
81            
82            <label for="video">Enviar Vídeo ({{ analysis_type|upper }} - {{ side|capfirst }}):</label>
83            <input type="file" id="video" name="video" accept=".mp4,.avi,.mov" required>
84            
85            <h3>Repetições para Salvar</h3>
86            <div class="checkbox-group">
87                <label><input type="checkbox" name="rep_1" value="1"> Repetição 1</label>
88                <label><input type="checkbox" name="rep_2" value="2"> Repetição 2</label>
89                <label><input type="checkbox" name="rep_3" value="3"> Repetição 3</label>
90            </div>
91            
92            <h3>Parâmetros de Avaliação</h3>
93            <div class="form-row">
94                <div class="form-col">
95                    <label for="descent_threshold">Sensibilidade da Descida:</label>
96                    <input type="range" id="descent_threshold" name="descent_threshold" 
97                           min="0.01" max="0.10" step="0.005" value="0.05">
98                    <span id="descent_value">0.05</span>
99                    <p class="help-text">Percentual de movimento da orelha para baixo.</p>
100                   
101                   <label for="hip_error_threshold">Tolerância Quadril:</label>
102                   <input type="range" id="hip_error_threshold" name="hip_error_threshold" 
103                          min="1" max="150" step="1" value="1">
104                   <span id="hip_value">1</span>
105                   <p class="help-text">Instantes permitidos de desvio do quadril.</p>
106               </div>
107               
108               <div class="form-col">
109                   <label for="ascent_return_threshold">Retorno na Subida:</label>
110                   <input type="range" id="ascent_return_threshold" name="ascent_return_threshold" 
111                          min="0.005" max="0.05" step="0.005" value="0.02">
112                   <span id="ascent_value">0.02</span>
113                   <p class="help-text">Proximidade da posição inicial.</p>
114                   
115                   <label for="knee_valgus_error_threshold">Tolerância Joelho:</label>
116                   <input type="range" id="knee_valgus_error_threshold" name="knee_valgus_error_threshold" 
117                          min="1" max="150" step="1" value="12">
118                   <span id="knee_value">12</span>
119                   <p class="help-text">Instantes permitidos de valgo/varo.</p>
120                   
121                   <label for="foot_pronation_error_threshold">Tolerância Pé:</label>
122                   <input type="range" id="foot_pronation_error_threshold" name="foot_pronation_error_threshold" 
123                          min="1" max="150" step="1" value="7">
124                   <span id="foot_value">7</span>
125                   <p class="help-text">Instantes permitidos de pronação.</p>
126               </div>
127           </div>
128           
129           <button type="submit" class="btn">Analisar Vídeo</button>
130       </form>
131   
132       <script>
133           // Atualiza valores dos sliders
134           document.querySelectorAll('input[type="range"]').forEach(slider => {
135               slider.addEventListener('input', function() {
136                   const span = document.getElementById(this.id.replace('_threshold', '_value'));
137                   if (span) span.textContent = this.value;
138               });
139           });
140       </script>
141   {% endif %}
142   {% endblock %}
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/squat_analyzer/templates/squat_analyzer/index.html
*Saved at: 23/06/2026, 14:54:56*

**[ADDED]**
```
1     {% extends 'squat_analyzer/base.html' %}
2     
3     {% block title %}Análise de Agachamento - Home{% endblock %}
4     
5     {% block content %}
6     <h1>Programa de Análise de Agachamento</h1>
7     <p style="margin-bottom: 30px;">Por favor, selecione o tipo de agachamento que você deseja analisar:</p>
8     
9     <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
10        <div class="card">
11            <h3>Agachamento Frontal</h3>
12            <p style="margin: 15px 0;">Análise do movimento frontal para detectar desvios laterais.</p>
13            <a href="{% url 'frontal_analysis' 'direito' %}" class="btn">Frontal Direito</a>
14            <a href="{% url 'frontal_analysis' 'esquerdo' %}" class="btn">Frontal Esquerdo</a>
15        </div>
16        
17        <div class="card">
18            <h3>Agachamento Sagital</h3>
19            <p style="margin: 15px 0;">Análise do movimento sagital para detectar desvios de perfil.</p>
20            <a href="{% url 'sagittal_analysis' 'direito' %}" class="btn">Sagital Direito</a>
21            <a href="{% url 'sagittal_analysis' 'esquerdo' %}" class="btn">Sagital Esquerdo</a>
22        </div>
23    </div>
24    {% endblock %}
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/squat_analyzer/templates/squat_analyzer/base.html
*Saved at: 23/06/2026, 14:54:06*

**[ADDED]**
```
1     <!DOCTYPE html>
2     <html lang="pt-BR">
3     <head>
4         <meta charset="UTF-8">
5         <meta name="viewport" content="width=device-width, initial-scale=1.0">
6         <title>{% block title %}Análise de Agachamento{% endblock %}</title>
7         <style>
8             * { margin: 0; padding: 0; box-sizing: border-box; }
9             body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
10            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
11            h1 { color: #333; margin-bottom: 20px; }
12            h2 { color: #555; margin: 20px 0 15px; }
13            h3 { color: #666; margin: 15px 0 10px; }
14            .btn { display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; border: none; cursor: pointer; font-size: 14px; }
15            .btn:hover { background: #0056b3; }
16            .btn-secondary { background: #6c757d; }
17            .btn-secondary:hover { background: #545b62; }
18            form { margin: 20px 0; }
19            label { display: block; margin: 15px 0 5px; font-weight: bold; }
20            input[type="text"], input[type="number"], input[type="file"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
21            input[type="range"] { width: 100%; }
22            .form-row { display: flex; gap: 20px; margin: 15px 0; }
23            .form-col { flex: 1; }
24            .checkbox-group { margin: 10px 0; }
25            .checkbox-group label { display: inline-block; margin-right: 20px; font-weight: normal; }
26            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
27            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
28            th { background: #f8f9fa; font-weight: bold; }
29            tr:hover { background: #f5f5f5; }
30            .alert { padding: 15px; margin: 15px 0; border-radius: 4px; }
31            .alert-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
32            .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
33            .feedback { background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }
34            .ok { color: #28a745; font-weight: bold; }
35            .error { color: #dc3545; font-weight: bold; }
36            .help-text { font-size: 12px; color: #666; margin-top: 5px; }
37            .card { border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 4px; }
38        </style>
39    </head>
40    <body>
41        <div class="container">
42            {% block content %}{% endblock %}
43        </div>
44    </body>
45    </html>
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/settings.py
*Saved at: 23/06/2026, 14:53:38*

**[REMOVED]**
```
(from line ~129)
FILE_UPLOAD_MAX_MEMORY_SIZE = 102428800  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 102428800  # 50MB

```
**[ADDED]**
```
129   FILE_UPLOAD_MAX_MEMORY_SIZE = 102428800  # 100MB
130   DATA_UPLOAD_MAX_MEMORY_SIZE = 102428800  # 100MB
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/settings.py
*Saved at: 23/06/2026, 14:51:15*

**[REMOVED]**
```
(from line ~129)
FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB

```
**[ADDED]**
```
129   FILE_UPLOAD_MAX_MEMORY_SIZE = 102428800  # 50MB
130   DATA_UPLOAD_MAX_MEMORY_SIZE = 102428800  # 50MB
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/settings.py
*Saved at: 23/06/2026, 14:50:56*

**[ADDED]**
```
119   
120   # Configurações de arquivos de mídia (uploads)
121   MEDIA_URL = '/media/'
122   MEDIA_ROOT = BASE_DIR / 'media'
123   
124   # Configurações de arquivos estáticos
125   STATIC_URL = 'static/'
126   STATICFILES_DIRS = [BASE_DIR / 'squat_analysis_app' / 'squat_analyzer' / 'static']
127   
128   # Tempo máximo para upload de vídeos grandes
129   FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB
130   DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/squat_analyzer/urls.py
*Saved at: 23/06/2026, 14:50:42*

**[ADDED]**
```
1     from django.urls import path
2     from . import views
3     
4     urlpatterns = [
5         path('', views.index, name='index'),
6         path('frontal/<str:side>/', views.frontal_analysis, name='frontal_analysis'),
7         path('sagittal/<str:side>/', views.sagittal_analysis, name='sagittal_analysis'),
8     ]
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/settings.py
*Saved at: 23/06/2026, 14:50:01*

**[ADDED]**
```
40        'squat_analyzer',
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/requirements.txt
*Saved at: 23/06/2026, 14:50:00*

**[REMOVED]**
```
(from line ~1)
# Django Framework
Django>=4.2,<5.0

```
**[ADDED]**
```
1     # Django framework
2     Django>=4.2
```
**[REMOVED]**
```
(from line ~4)
# Django REST Framework (opcional, para APIs futuras)
djangorestframework>=3.14.0

```
**[ADDED]**
```
4     # Processamento de vídeo e ML (mantidos do Streamlit)
5     opencv-python
6     mediapipe
7     numpy
8     pandas
```
**[REMOVED]**
```
(from line ~10)
# Processamento de dados e cálculos
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0

```
**[ADDED]**
```
10    # Geração de relatórios Excel
11    openpyxl
```
**[REMOVED]**
```
(from line ~13)
# Processamento de vídeo e visão computacional
opencv-python>=4.8.0
mediapipe>=0.10.0

```
**[ADDED]**
```
13    # Modelos de Machine Learning
14    scikit-learn
15    joblib
```
**[REMOVED]**
```
(from line ~17)
# Manipulação de arquivos Excel
openpyxl>=3.1.0

# Variáveis de ambiente
python-decouple>=3.8

# Pillow para manipulação de imagens
Pillow>=10.0.0

# Utilitários
python-dateutil>=2.8.0
```
**[ADDED]**
```
17    # Navegador para testes (opcional)
18    webdriver-manager
19    selenium
```

---

### 📄 /home/piva/Documentos/Programação/Python/Programa Agachamento/squat_analysis_app/urls.py
*Saved at: 23/06/2026, 14:49:59*

**[REMOVED]**
```
(from line ~1)
"""
URL configuration for squat_analysis_app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

```
**[REMOVED]**
```
(from line ~2)
from django.urls import path

```
**[ADDED]**
```
2     from django.urls import path, include
3     from django.conf import settings
4     from django.conf.urls.static import static
```
**[ADDED]**
```
8         path('', include('squat_analyzer.urls')),  # ← Incluir URLs do app
```
**[ADDED]**
```
10    
11    # Permitir upload de arquivos de mídia
12    if settings.DEBUG:
13        urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

