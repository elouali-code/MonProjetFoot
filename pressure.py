import numpy as np
import cv2

class PressureMap:
    def __init__(self, pitch_width=105, pitch_height=68, resolution=1.0):
        """
        pitch_width, pitch_height : Dimensions réelles en mètres
        resolution : 1.0 = 1 pixel pour 1 mètre (rapide). 0.5 = 2 pixels par mètre (précis).
        """
        self.w = int(pitch_width * resolution)
        self.h = int(pitch_height * resolution)
        self.res = resolution
        
        # Création de la grille de coordonnées (Meshgrid)
        # x_grid et y_grid contiennent les coordonnées de chaque pixel de la carte
        x = np.linspace(0, pitch_width, self.w)
        y = np.linspace(0, pitch_height, self.h)
        self.xv, self.yv = np.meshgrid(x, y)

    def gaussian_influence(self, mx, my, sigma=4.0):
        """
        Calcule une carte d'influence pour UN joueur situé en (mx, my).
        Formule : exp( -dist^2 / (2*sigma^2) )
        """
        # Calcul vectorisé de la distance au carré
        dist_sq = (self.xv - mx)**2 + (self.yv - my)**2
        
        # Application de la gaussienne
        influence = np.exp(-dist_sq / (2 * sigma**2))
        return influence

    def generate_heatmap(self, players, team_of_interest_id=None):
        """
        Génère la heatmap de pression totale.
        players : Liste de tuples [(x, y, team_id), ...]
        team_of_interest_id : Si spécifié, on calcule seulement la pression exercée PAR cette équipe.
        """
        # Carte vide (zéros)
        total_pressure = np.zeros((self.h, self.w))
        
        for p in players:
            px, py, team_id = p
            
            # Filtre : Si on veut voir la pression du Maroc, on ignore le Nigeria
            if team_of_interest_id is not None and team_id != team_of_interest_id:
                continue
                
            # Vérification basique pour ne pas calculer des fantômes hors terrain
            if px < 0 or px > (self.w/self.res) or py < 0 or py > (self.h/self.res):
                continue
            
            # Ajout de l'influence du joueur à la carte globale
            # Sigma=5.0 mètres (rayon d'influence moyen)
            total_pressure += self.gaussian_influence(px, py, sigma=5.0)

        # Normalisation pour l'affichage (0 à 1)
        if np.max(total_pressure) > 0:
            total_pressure = total_pressure / np.max(total_pressure)
        
        return total_pressure

    def visualize(self, pressure_grid, width_display, height_display):
        """
        Transforme la grille mathématique en image couleur (Heatmap)
        """
        # Convertir en format image 0-255
        heatmap_uint8 = np.uint8(255 * pressure_grid)
        
        # Appliquer une colormap (JET = Bleu froid -> Rouge chaud)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Redimensionner à la taille d'affichage voulue
        return cv2.resize(heatmap_color, (width_display, height_display))