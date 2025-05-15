import cv2
import numpy as np
import dlib
import os
import pickle
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from skimage.feature import local_binary_pattern
from pathlib import Path

class EnhancedFacialRecognitionSystem:
    """
    Système avancé de reconnaissance faciale basé sur HOG et LBP
    Stocke uniquement les vecteurs de caractéristiques, pas les images
    Avec contrôle d'accès par département et détection d'intrus
    """
    
    def __init__(self, 
                 similarity_threshold=0.92,  # Increased from 0.78 to reduce false positives
                 confidence_levels=20,
                 enable_lbp=True,
                 hog_cell_size=(8, 8),
                 hog_block_size=(2, 2),
                 use_average_update=True,
                 database_path="face_vectors_db",
                 intruder_folder="intruder_captures"):
        """
        Initialisation du système de reconnaissance faciale
        
        Args:
            similarity_threshold (float): Seuil de similarité pour considérer une correspondance
            confidence_levels (int): Nombre de vecteurs à stocker par personne pour différentes expressions/angles
            enable_lbp (bool): Activer la combinaison avec LBP pour plus de robustesse
            hog_cell_size (tuple): Taille des cellules HOG
            hog_block_size (tuple): Taille des blocs HOG
            use_average_update (bool): Utiliser la moyenne pondérée pour les mises à jour
            database_path (str): Chemin vers le dossier de stockage de la base de données
            intruder_folder (str): Dossier pour stocker les captures d'intrus
        """
        # Configuration des paramètres
        self.similarity_threshold = similarity_threshold
        self.confidence_levels = confidence_levels
        self.enable_lbp = enable_lbp
        self.use_average_update = use_average_update
        self.database_path = database_path
        self.intruder_folder = intruder_folder
        
        # Création des dossiers nécessaires
        Path(self.database_path).mkdir(parents=True, exist_ok=True)
        Path(self.intruder_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialisation des détecteurs et prédicteurs
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Chemin vers le fichier du prédicteur de points faciaux
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        # Téléchargement automatique du fichier si nécessaire
        if not os.path.exists(predictor_path):
            print(f"Le fichier {predictor_path} n'existe pas, vous devez le télécharger.")
            print("Vous pouvez le télécharger depuis: https://github.com/davisking/dlib-models")
        
        self.landmark_predictor = dlib.shape_predictor(predictor_path)
        
        # Configuration du descripteur HOG optimisé pour les visages
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(hog_block_size[0] * hog_cell_size[0], 
                       hog_block_size[1] * hog_cell_size[1]),
            _blockStride=(hog_cell_size[0] // 2, hog_cell_size[1] // 2),
            _cellSize=hog_cell_size,
            _nbins=9,
            _derivAperture=1,
            _winSigma=-1,
            _histogramNormType=0,
            _L2HysThreshold=0.2,
            _gammaCorrection=True,
            _nlevels=64,
            _signedGradient=False)
        
        # Paramètres LBP
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        
        # Dictionnaire pour stocker les vecteurs de référence {id: [liste_vecteurs_hog]}
        self.face_database = {}
        
        # Chargement de la base de données si elle existe
        self.load_database()
        
        # Stockage des métadonnées pour chaque personne
        self.metadata = {}
        
        # Liste des départements disponibles
        self.departments = ["Administration", "Production", "RH", "Finance", "R&D", "IT", "Sécurité", "Maintenance", "Logistique", "Accueil"]
        
        # Journalisation
        self.log_file = os.path.join(self.database_path, "recognition_log.txt")
        self.intruder_log = os.path.join(self.database_path, "intruder_log.txt")
        
        # Compteur pour les intrus détectés
        self.intruder_counter = self._load_intruder_counter()
    
    def _load_intruder_counter(self):
        """Charge le compteur d'intrus depuis un fichier"""
        counter_file = os.path.join(self.database_path, "intruder_counter.txt")
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                try:
                    return int(f.read().strip())
                except:
                    return 0
        return 0
    
    def _save_intruder_counter(self):
        """Sauvegarde le compteur d'intrus dans un fichier"""
        counter_file = os.path.join(self.database_path, "intruder_counter.txt")
        with open(counter_file, 'w') as f:
            f.write(str(self.intruder_counter))
    
    def preprocess_face(self, image, face_rect=None):
        """
        Prétraite l'image du visage pour normaliser l'éclairage et l'orientation
        
        Args:
            image: Image contenant un visage
            face_rect: Rectangle de détection de visage (si déjà détecté)
            
        Returns:
            Tuple (visage prétraité, rectangle facial)
        """
        # Conversion en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Détection du visage si non fourni
        if face_rect is None:
            faces = self.face_detector(gray)
            if not faces:
                return None, None
            face_rect = faces[0]
            
        try:
            # Extraction des points faciaux
            landmarks = self.landmark_predictor(gray, face_rect)
            
            # Calcul des points centraux des yeux
            left_eye = np.mean(np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                        for i in range(36, 42)]), axis=0)
            right_eye = np.mean(np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                         for i in range(42, 48)]), axis=0)
            
            # Calcul de l'angle pour l'alignement des yeux
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Extraction des coordonnées du rectangle facial
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
            
            # Ajout d'une marge autour du visage (20%)
            margin_x = int(0.2 * w)
            margin_y = int(0.2 * h)
            x_min = max(0, x - margin_x)
            y_min = max(0, y - margin_y)
            x_max = min(gray.shape[1], x + w + margin_x)
            y_max = min(gray.shape[0], y + h + margin_y)
            
            # Extraction du visage avec marge
            face_region = gray[y_min:y_max, x_min:x_max]
            
            # Calcul du centre de rotation
            center = ((x_max - x_min) // 2, (y_max - y_min) // 2)
            
            # Matrice de rotation
            M = cv2.getRotationMatrix2D(center, angle, 1)
            
            # Application de la rotation
            rotated_face = cv2.warpAffine(face_region, M, (face_region.shape[1], face_region.shape[0]))
            
            # Égalisation d'histogramme adaptative pour normaliser l'éclairage
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized_face = clahe.apply(rotated_face)
            
            # Redimensionnement pour normalisation (taille fixe pour HOG)
            face_resized = cv2.resize(normalized_face, (64, 128))
            
            return face_resized, face_rect
        except Exception as e:
            print(f"Erreur lors du prétraitement: {e}")
            return None, None
    
    def extract_features(self, processed_face):
        """
        Extrait les caractéristiques HOG et LBP du visage
        
        Args:
            processed_face: Visage prétraité
            
        Returns:
            Vecteur de caractéristiques combiné
        """
        # Calcul des caractéristiques HOG
        hog_features = self.hog.compute(processed_face)
        
        # Normalisation du vecteur HOG
        hog_features = normalize(hog_features.reshape(1, -1))[0]
        
        # Si LBP est activé, on combine les caractéristiques
        if self.enable_lbp:
            # Calcul des caractéristiques LBP
            lbp = local_binary_pattern(processed_face, self.lbp_n_points, self.lbp_radius, method='uniform')
            
            # Création d'un histogramme des motifs LBP
            n_bins = self.lbp_n_points + 2
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            
            # Normalisation de l'histogramme LBP
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            # Concaténation des caractéristiques HOG et LBP
            combined_features = np.concatenate([hog_features, lbp_hist])
            
            # Normalisation finale
            combined_features = normalize(combined_features.reshape(1, -1))[0]
            
            return combined_features
        
        return hog_features
    
    def register_face(self, image, person_id, metadata=None):
        """
        Enregistre un nouveau visage dans la base de données
        
        Args:
            image: Image contenant un visage
            person_id: Identifiant unique de la personne
            metadata: Métadonnées supplémentaires (dictionnaire)
            
        Returns:
            Booléen indiquant si l'enregistrement a réussi
        """
        # Prétraitement du visage
        processed_face, face_rect = self.preprocess_face(image)
        
        if processed_face is None:
            return False
        
        # Extraction des caractéristiques
        features = self.extract_features(processed_face)
        
        # Vérifier si cette personne existe déjà et si le nouveau visage est trop similaire
        # avec des personnes existantes autres que person_id
        if len(self.face_database) > 0:
            # Vérifier la similarité avec toutes les autres personnes
            for existing_id, feature_vectors in self.face_database.items():
                if existing_id != person_id:  # Ignorer la personne actuelle
                    for stored_features in feature_vectors:
                        similarity = cosine_similarity(features.reshape(1, -1), 
                                                     stored_features.reshape(1, -1))[0][0]
                        
                        # Si le nouveau visage est trop similaire à un visage existant d'une autre personne
                        if similarity > self.similarity_threshold - 0.05:  # Une marge de tolérance
                            print(f"AVERTISSEMENT: Ce visage est très similaire à {existing_id} (score: {similarity:.4f})")
                            # Vous pouvez choisir de rejeter l'enregistrement ici en décommentant la ligne suivante
                            # return False
        
        # Création ou mise à jour de l'entrée dans la base de données
        if person_id not in self.face_database:
            self.face_database[person_id] = []
            self.metadata[person_id] = {
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vector_count": 0,
                "recognition_count": 0,
                "department_access": [],  # Nouveau: liste des départements accessibles
                "status": "authorized"    # Nouveau: état d'autorisation (authorized/revoked)
            }
            
            # Ajout des métadonnées fournies
            if metadata:
                self.metadata[person_id].update(metadata)
        
        # Mise à jour de la date de dernière modification
        self.metadata[person_id]["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Limitation du nombre de vecteurs par personne
        if len(self.face_database[person_id]) >= self.confidence_levels:
            # On remplace le vecteur le plus ancien
            self.face_database[person_id].pop(0)
        
        # Ajout du nouveau vecteur
        self.face_database[person_id].append(features)
        self.metadata[person_id]["vector_count"] = len(self.face_database[person_id])
        
        # Sauvegarde de la base de données
        self.save_database()
        
        return True
    
    def set_department_access(self, person_id, departments):
        """
        Définit les départements auxquels une personne a accès
        
        Args:
            person_id: Identifiant de la personne
            departments: Liste des départements autorisés
            
        Returns:
            Booléen indiquant si la mise à jour a réussi
        """
        if person_id not in self.metadata:
            return False
        
        # Validation des départements
        valid_departments = [dept for dept in departments if dept in self.departments]
        
        # Mise à jour des accès
        self.metadata[person_id]["department_access"] = valid_departments
        self.save_database()
        
        return True
    
    def check_department_access(self, person_id, department):
        """
        Vérifie si une personne a accès à un département spécifique
        
        Args:
            person_id: Identifiant de la personne
            department: Département à vérifier
            
        Returns:
            Booléen indiquant si l'accès est autorisé
        """
        if person_id not in self.metadata:
            return False
        
        # Vérifier le statut d'autorisation global
        if self.metadata[person_id].get("status", "authorized") != "authorized":
            return False
        
        # Accès super administrateur à tous les départements
        if "Tous" in self.metadata[person_id].get("department_access", []):
            return True
        
        # Vérifier l'accès au département spécifique
        return department in self.metadata[person_id].get("department_access", [])
    
    def recognize_face(self, image, return_all_matches=False, min_matches=1, min_confidence=0.92):
        """
        Reconnait un visage par comparaison avec la base de données
        
        Args:
            image: Image contenant un visage
            return_all_matches: Retourner toutes les correspondances au-dessus du seuil
            min_matches: Nombre minimum de vecteurs qui doivent correspondre
            min_confidence: Confiance minimum absolue pour considérer une correspondance valide
            
        Returns:
            Identité reconnue (ou liste des correspondances) et score de confiance
        """
        # Prétraitement du visage
        processed_face, face_rect = self.preprocess_face(image)
        
        if processed_face is None:
            return None, 0.0
        
        # Extraction des caractéristiques
        features = self.extract_features(processed_face)
        
        # Dictionnaire des correspondances {person_id: scores}
        matches = {}
        
        # Comparaison avec tous les visages enregistrés
        for person_id, feature_vectors in self.face_database.items():
            person_scores = []
            
            # Comparaison avec tous les vecteurs de la personne
            for stored_features in feature_vectors:
                similarity = cosine_similarity(features.reshape(1, -1), 
                                             stored_features.reshape(1, -1))[0][0]
                person_scores.append(similarity)
            
            # Calcul du score moyen pour cette personne
            if len(person_scores) >= min_matches:
                # On prend les X meilleurs scores
                top_scores = sorted(person_scores, reverse=True)[:min_matches]
                avg_score = sum(top_scores) / len(top_scores)
                
                # Application d'un seuil absolu en plus du seuil relatif
                if avg_score >= self.similarity_threshold and avg_score >= min_confidence:
                    matches[person_id] = avg_score
        
        # Journalisation de la reconnaissance
        self._log_recognition(matches)
        
        # Si aucune correspondance, enregistrer comme intrus
        if not matches:
            self._capture_intruder(image, face_rect)
            return None, 0.0
        
        # Trier les correspondances par score décroissant
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        # NOUVEAU: Vérifier l'écart entre la meilleure correspondance et la suivante
        if len(sorted_matches) > 1:
            best_score = sorted_matches[0][1]
            second_score = sorted_matches[1][1]
            
            # Si les deux meilleurs scores sont trop proches, considérer comme ambigu
            if best_score - second_score < 0.05:  # Seuil d'ambiguïté
                print(f"AVERTISSEMENT: Reconnaissance ambiguë - {sorted_matches[0][0]} ({best_score:.4f}) vs {sorted_matches[1][0]} ({second_score:.4f})")
                # Option alternative: retourner None en cas d'ambiguïté
                # return None, 0.0
        
        # Retourner toutes les correspondances ou juste la meilleure
        if return_all_matches:
            return sorted_matches, sorted_matches[0][1]
        else:
            best_match = sorted_matches[0]
            
            # Mettre à jour le compteur de reconnaissance
            if best_match[0] in self.metadata:
                self.metadata[best_match[0]]["recognition_count"] += 1
                self.save_database()
                
            return best_match[0], best_match[1]
    
    def _capture_intruder(self, image, face_rect):
        """
        Enregistre une capture d'écran d'un intrus détecté
        
        Args:
            image: Image contenant l'intrus
            face_rect: Rectangle du visage détecté
        """
        if face_rect is None:
            return
        
        # Incrémenter le compteur d'intrus
        self.intruder_counter += 1
        self._save_intruder_counter()
        
        # Nom de fichier avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intruder_{self.intruder_counter}_{timestamp}.jpg"
        filepath = os.path.join(self.intruder_folder, filename)
        
        # Extraction du visage avec une marge
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        
        # Ajout d'une marge autour du visage (50%)
        margin_x = int(0.5 * w)
        margin_y = int(0.5 * h)
        x_min = max(0, x - margin_x)
        y_min = max(0, y - margin_y)
        x_max = min(image.shape[1], x + w + margin_x)
        y_max = min(image.shape[0], y + h + margin_y)
        
        # Extraction et sauvegarde de l'image
        try:
            face_img = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(filepath, face_img)
            
            # Journalisation de l'intrus
            with open(self.intruder_log, 'a') as f:
                log_entry = f"{timestamp} - Intrus détecté et capturé: {filename}\n"
                f.write(log_entry)
                
            print(f"ALERTE: Intrus détecté et capturé ({filename})")
        except Exception as e:
            print(f"Erreur lors de la capture de l'intrus: {e}")
    
    def update_face_vector(self, person_id, image, weight=0.3):
        """
        Met à jour le vecteur facial d'une personne existante
        
        Args:
            person_id: Identifiant de la personne
            image: Nouvelle image du visage
            weight: Poids du nouveau vecteur (entre 0 et 1)
            
        Returns:
            Booléen indiquant si la mise à jour a réussi
        """
        if person_id not in self.face_database:
            return False
        
        # Prétraitement du visage
        processed_face, face_rect = self.preprocess_face(image)
        
        if processed_face is None:
            return False
        
        # Extraction des caractéristiques
        new_features = self.extract_features(processed_face)
        
        # Vérifier si cette mise à jour ne crée pas de confusion avec d'autres personnes
        for existing_id, feature_vectors in self.face_database.items():
            if existing_id != person_id:  # Ignorer la personne actuelle
                for stored_features in feature_vectors:
                    similarity = cosine_similarity(new_features.reshape(1, -1), 
                                                 stored_features.reshape(1, -1))[0][0]
                    
                    # Si le nouveau visage est trop similaire à un visage existant d'une autre personne
                    if similarity > self.similarity_threshold - 0.05:  # Une marge de tolérance
                        print(f"AVERTISSEMENT: Ce visage est très similaire à {existing_id} (score: {similarity:.4f})")
                        # Vous pouvez choisir de rejeter la mise à jour ici en décommentant la ligne suivante
                        # return False
        
        if self.use_average_update:
            # Sélection du vecteur le plus similaire pour la mise à jour
            max_similarity = -1
            update_index = 0
            
            for i, stored_features in enumerate(self.face_database[person_id]):
                similarity = cosine_similarity(new_features.reshape(1, -1), 
                                             stored_features.reshape(1, -1))[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    update_index = i
            
            # Moyenne pondérée entre l'ancien et le nouveau vecteur
            old_vector = self.face_database[person_id][update_index]
            updated_vector = (1 - weight) * old_vector + weight * new_features
            
            # Normalisation
            updated_vector = normalize(updated_vector.reshape(1, -1))[0]
            
            # Mise à jour du vecteur
            self.face_database[person_id][update_index] = updated_vector
            
        else:
            # Ajout d'un nouveau vecteur (avec limitation)
            if len(self.face_database[person_id]) >= self.confidence_levels:
                # On remplace le vecteur le plus ancien
                self.face_database[person_id].pop(0)
            
            # Ajout du nouveau vecteur
            self.face_database[person_id].append(new_features)
        if person_id not in self.metadata:
            self.metadata[person_id] = {
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_update": None,
                "vector_count": 0,
                "recognition_count": 0,
                "department_access": [],
                "status": "authorized"
            }
        
        # Mise à jour des métadonnées
        self.metadata[person_id]["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata[person_id]["vector_count"] = len(self.face_database[person_id])
        
        # Sauvegarde de la base de données
        self.save_database()
        
        return True
    
    def save_database(self):
        """Sauvegarde la base de données des vecteurs faciaux"""
        db_file = os.path.join(self.database_path, "face_vectors.pkl")
        metadata_file = os.path.join(self.database_path, "metadata.json")
        
        # Sauvegarde des vecteurs (binaire)
        with open(db_file, 'wb') as f:
            pickle.dump(self.face_database, f)
        
        # Sauvegarde des métadonnées (JSON)
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def load_database(self):
        """Charge la base de données des vecteurs faciaux"""
        db_file = os.path.join(self.database_path, "face_vectors.pkl")
        metadata_file = os.path.join(self.database_path, "metadata.json")
        
        # Chargement des vecteurs
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    self.face_database = pickle.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de la base de données: {e}")
                print("Création d'une nouvelle base de données.")
                self.face_database = {}
        
        # Chargement des métadonnées
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement des métadonnées: {e}")
                print("Création de nouvelles métadonnées.")
                self.metadata = {}
        
        # Vérification de cohérence entre face_database et metadata
        for person_id in list(self.face_database.keys()):
            if person_id not in self.metadata:
                print(f"Métadonnées manquantes pour {person_id}, création automatique.")
                self.metadata[person_id] = {
                    "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "vector_count": len(self.face_database[person_id]),
                    "recognition_count": 0,
                    "department_access": [],
                    "status": "authorized"
                }
    
    def delete_person(self, person_id):
        """
        Supprime une personne de la base de données
        
        Args:
            person_id: Identifiant de la personne à supprimer
            
        Returns:
            Booléen indiquant si la suppression a réussi
        """
        if person_id in self.face_database:
            del self.face_database[person_id]
            
            if person_id in self.metadata:
                del self.metadata[person_id]
            
            self.save_database()
            return True
        
        return False
    
    def get_database_info(self):
        """
        Retourne des informations sur la base de données
        
        Returns:
            Dictionnaire contenant les informations de la base de données
        """
        return {
            "total_persons": len(self.face_database),
            "total_vectors": sum(len(vectors) for vectors in self.face_database.values()),
            "persons": self.metadata
        }
    
    def _log_recognition(self, matches):
        """Journalise les événements de reconnaissance"""
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if matches:
                top_match = sorted(matches.items(), key=lambda x: x[1], reverse=True)[0]
                log_entry = f"{timestamp} - Reconnaissance: {top_match[0]} (confiance: {top_match[1]:.4f})\n"
            else:
                log_entry = f"{timestamp} - Aucune correspondance trouvée\n"
            f.write(log_entry)
    
    def optimize_database(self, duplicates_threshold=0.95):
        """
        Optimise la base de données en fusionnant les vecteurs très similaires
        
        Args:
            duplicates_threshold: Seuil pour considérer des vecteurs comme duplicats
            
        Returns:
            Nombre de vecteurs supprimés
        """
        total_removed = 0
        
        for person_id, vectors in self.face_database.items():
            # Copie des vecteurs pour éviter les problèmes de modification pendant l'itération
            vectors_copy = vectors.copy()
            
            # Recherche de vecteurs similaires
            i = 0
            while i < len(vectors_copy):
                j = i + 1
                while j < len(vectors_copy):
                    similarity = cosine_similarity(vectors_copy[i].reshape(1, -1),
                                                 vectors_copy[j].reshape(1, -1))[0][0]
                    
                    if similarity > duplicates_threshold:
                        # Fusion des vecteurs (moyenne)
                        merged_vector = (vectors_copy[i] + vectors_copy[j]) / 2
                        merged_vector = normalize(merged_vector.reshape(1, -1))[0]
                        
                        # Remplacement du premier vecteur
                        vectors_copy[i] = merged_vector
                        
                        # Suppression du second vecteur
                        vectors_copy.pop(j)
                        total_removed += 1
                    else:
                        j += 1
                i += 1
                if person_id not in self.metadata:
                    self.metadata[person_id] = {
                        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "last_update": None,
                        "vector_count": 0,
                        "recognition_count": 0
                    }
            # Mise à jour de la base de données
            self.face_database[person_id] = vectors_copy
            self.metadata[person_id]["vector_count"] = len(vectors_copy)
        
        # Sauvegarde des modifications
        if total_removed > 0:
            self.save_database()
        
        return total_removed
    
    def reset_database(self):
        """
        Réinitialise complètement la base de données
        
        Returns:
            Booléen indiquant si la réinitialisation a réussi
        """
        try:
            self.face_database = {}
            self.metadata = {}
            self.save_database()
            
            print("Base de données réinitialisée avec succès.")
            return True
        except Exception as e:
            print(f"Erreur lors de la réinitialisation de la base de données: {e}")
            return False
    
    def verify_face_pair(self, image1, image2):
        """
        Vérifie si deux images contiennent la même personne
        
        Args:
            image1: Première image
            image2: Deuxième image
            
        Returns:
            Tuple (booléen indiquant si c'est la même personne, score de similarité)
        """
        # Prétraitement des visages
        processed_face1, _ = self.preprocess_face(image1)
        processed_face2, _ = self.preprocess_face(image2)
        
        if processed_face1 is None or processed_face2 is None:
            return False, 0.0
        
        # Extraction des caractéristiques
        features1 = self.extract_features(processed_face1)
        features2 = self.extract_features(processed_face2)
        
        # Calcul de la similarité
        similarity = cosine_similarity(features1.reshape(1, -1), 
                                     features2.reshape(1, -1))[0][0]
        
        # Vérification du seuil
        return similarity >= self.similarity_threshold, similarity


# Application de démonstration du système de reconnaissance faciale
def demo_recognition_system():
    """
    Démontre l'utilisation du système de reconnaissance faciale avancé
    Version modifiée pour afficher les prompts dans la fenêtre et non dans la console
    Avec ajout d'authentification et message de bienvenue
    """
    import time
    
    # Initialisation du système
    system = EnhancedFacialRecognitionSystem(
        similarity_threshold=0.92,  # Augmenté pour réduire les faux positifs
        confidence_levels=20,
        enable_lbp=True,
        hog_cell_size=(8, 8),
        hog_block_size=(2, 2)
    )
    
    # Capture vidéo (0 pour la webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam")
        return
    
    # Mode actuel (reconnaissance ou enregistrement)
    mode = "recognition"
    current_id = None
    
    # Variables pour gérer les prompts et les entrées
    input_active = False
    prompt_text = ""
    user_input = ""
    operation_feedback = ""
    feedback_timer = 0
    
    # Variables pour l'authentification
    authenticated_person = None
    auth_confidence = 0.0
    auth_timestamp = 0
    welcome_displayed = False
    auth_department = ""  # Pour stocker le département actuellement accédé
    
    # Fonction pour gérer les clics de souris
    def mouse_callback(event, x, y, flags, param):
        nonlocal input_active
        # Si l'utilisateur clique, on désactive la saisie
        if event == cv2.EVENT_LBUTTONDOWN and input_active:
            input_active = False
    
    # Création d'une window nommée pour pouvoir y attacher le callback
    cv2.namedWindow("Reconnaissance Faciale")
    cv2.setMouseCallback("Reconnaissance Faciale", mouse_callback)
    
    print("=== Système de reconnaissance faciale ===")
    
    while True:
        # Capture d'une image
        ret, frame = cap.read()
        
        if not ret:
            print("Erreur: Impossible de capturer une image")
            break
        
        # Sauvegarde de la frame actuelle pour l'enregistrement
        current_frame = frame.copy()
        
        # Miroir horizontal pour une utilisation plus intuitive
        frame = cv2.flip(frame, 1)
        
        # Copie pour l'affichage
        display = frame.copy()
        
        # Détection des visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = system.face_detector(gray)
        
        # Authentification périodique en mode reconnaissance
        if mode == "recognition" and len(faces) == 1:
            # Vérifier si c'est un nouveau visage ou si on doit réauthentifier
            current_time = time.time()
            
            # Réauthentification toutes les 3 secondes ou si nous n'avons pas encore authentifié
            if authenticated_person is None or current_time - auth_timestamp > 3:
                identity, confidence = system.recognize_face(frame)
                
                if identity and confidence > 0.92:  # Seuil d'authentification
                    # Personne reconnue avec une confiance suffisante
                    authenticated_person = identity
                    auth_confidence = confidence
                    auth_timestamp = current_time
                    welcome_displayed = False  # Réinitialiser pour afficher à nouveau le message
                else:
                    # Personne non reconnue ou confiance insuffisante
                    authenticated_person = None
        
        # Dessin des rectangles de visage
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Couleur selon le mode et l'authentification
            if mode == "recognition":
                if authenticated_person:
                    color = (0, 255, 0)  # Vert pour authentifié
                else:
                    color = (0, 165, 255)  # Orange pour non authentifié
            elif mode == "registration":
                color = (0, 0, 255)  # Rouge
            else:  # update
                color = (255, 0, 0)  # Bleu
            
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            
            # En mode reconnaissance, on affiche l'identité reconnue
            if mode == "recognition":
                if authenticated_person:
                    text = f"{authenticated_person} ({auth_confidence:.2f})"
                    cv2.putText(display, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Vérifier le statut d'autorisation
                    if authenticated_person in system.metadata:
                        person_status = system.metadata[authenticated_person].get("status", "authorized")
                        if person_status != "authorized":
                            status_color = (0, 0, 255)  # Rouge pour non autorisé
                            cv2.putText(display, "ACCÈS RÉVOQUÉ", (x, y+h+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                else:
                    cv2.putText(display, "Non authentifié", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Affichage du message de bienvenue si authentifié et message pas encore affiché
        if authenticated_person and not welcome_displayed:
            # Création d'un overlay pour le message de bienvenue
            welcome_overlay = display.copy()
            
            # Rectangle semi-transparent pour le fond du message
            cv2.rectangle(welcome_overlay, 
                         (50, display.shape[0]//2 - 100), 
                         (display.shape[1]-50, display.shape[0]//2 + 100), 
                         (0, 0, 0), -1)
            
            # Création du message de bienvenue
            welcome_message = f"Bienvenue, {authenticated_person}!"
            
            # Informations supplémentaires si disponibles dans les métadonnées
            if authenticated_person in system.metadata:
                person_data = system.metadata[authenticated_person]
                
                # Vérifier le statut d'autorisation
                status = person_data.get("status", "authorized")
                if status == "authorized":
                    status_message = "Statut: Autorisé"
                    status_color = (0, 255, 0)  # Vert
                    
                    # Afficher les départements accessibles
                    departments = person_data.get("department_access", [])
                    if departments:
                        dept_message = "Accès: " + ", ".join(departments)
                    else:
                        dept_message = "Aucun accès départemental configuré"
                else:
                    status_message = "Statut: ACCÈS RÉVOQUÉ"
                    status_color = (0, 0, 255)  # Rouge
                    dept_message = "Contactez l'administrateur système"
                
                # Nombre de reconnaissances
                recognition_count = person_data.get("recognition_count", 0)
                count_message = f"Reconnaissances: {recognition_count}"
                
                # Date de dernière mise à jour
                last_update = person_data.get("last_update", "Inconnue")
                update_message = f"Dernière mise à jour: {last_update}"
            else:
                status_message = "Statut: Information non disponible"
                status_color = (0, 165, 255)  # Orange
                dept_message = ""
                count_message = ""
                update_message = ""
            
            # Fusionner l'overlay avec l'affichage principal (transparence 70%)
            cv2.addWeighted(welcome_overlay, 0.7, display, 0.3, 0, display)
            
            # Afficher les messages
            cv2.putText(display, welcome_message, 
                       (70, display.shape[0]//2 - 60), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.putText(display, status_message, 
                       (70, display.shape[0]//2 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            if dept_message:
                cv2.putText(display, dept_message, 
                           (70, display.shape[0]//2 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            if count_message:
                cv2.putText(display, count_message, 
                           (70, display.shape[0]//2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            if update_message:
                cv2.putText(display, update_message, 
                           (70, display.shape[0]//2 + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Définir comme affiché
            welcome_displayed = True
        
        # Affichage des commandes disponibles
        cv2.putText(display, "R: Reconnaissance | E: Enregistrer | U: Mettre a jour | Q: Quitter | I: Info | D: Supprimer | O: Optimiser | A: Accès", 
                   (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Affichage du mode actuel
        cv2.putText(display, f"Mode: {mode}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Si en mode enregistrement ou mise à jour, afficher l'ID actuel
        if mode in ["registration", "update"] and current_id:
            cv2.putText(display, f"ID: {current_id}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Affichage du feedback d'opération
        if operation_feedback and time.time() - feedback_timer < 5:  # Afficher pendant 5 secondes
            cv2.putText(display, operation_feedback, 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            operation_feedback = ""
        
        # Gestion des prompts et entrées utilisateur
        if input_active:
            # Affichage du fond pour le prompt
            overlay = display.copy()
            cv2.rectangle(overlay, (50, display.shape[0]//2 - 40), 
                         (display.shape[1]-50, display.shape[0]//2 + 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            # Affichage du prompt et de l'entrée utilisateur
            cv2.putText(display, prompt_text, 
                       (60, display.shape[0]//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, user_input + "_", 
                       (60, display.shape[0]//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Affichage
        cv2.imshow("Reconnaissance Faciale", display)
        
        # Attente d'une touche
        key = cv2.waitKey(1) & 0xFF
        
        # Gestion des entrées pour les prompts actifs
        if input_active:
            if key == 13:  # Touche ENTER
                input_active = False
                # Traitement de l'entrée selon le contexte
                if prompt_text == "Entrez l'ID de la nouvelle personne:":
                    current_id = user_input
                    mode = "registration"
                    operation_feedback = f"Mode enregistrement activé pour ID: {current_id}"
                    feedback_timer = time.time()
                elif prompt_text == "Entrez l'ID de la personne à mettre à jour:":
                    if user_input in system.face_database:
                        current_id = user_input
                        mode = "update"
                        operation_feedback = f"Mode mise à jour activé pour ID: {current_id}"
                        feedback_timer = time.time()
                    else:
                        operation_feedback = f"Erreur: {user_input} n'existe pas dans la base de données"
                        feedback_timer = time.time()
                        mode = "recognition"
                        current_id = None
                elif prompt_text == "Entrez l'ID de la personne à supprimer:":
                    if system.delete_person(user_input):
                        operation_feedback = f"Personne {user_input} supprimée avec succès"
                        feedback_timer = time.time()
                    else:
                        operation_feedback = f"Erreur: {user_input} n'existe pas dans la base de données"
                        feedback_timer = time.time()
                elif prompt_text == "Entrez l'ID de la personne pour configurer l'accès:":
                    person_id = user_input
                    if person_id in system.metadata:
                        # Passage à la sélection du département
                        prompt_text = f"Départements (séparés par des virgules): {', '.join(system.departments)}"
                        input_active = True
                        user_input = ", ".join(system.metadata[person_id].get("department_access", []))
                    else:
                        operation_feedback = f"Erreur: {person_id} n'existe pas dans la base de données"
                        feedback_timer = time.time()
                        input_active = False
                elif prompt_text.startswith("Départements (séparés par des virgules):"):
                    # Récupération de l'ID depuis le feedback
                    person_id = current_id
                    if person_id in system.metadata:
                        # Traitement de la liste des départements
                        departments = [dept.strip() for dept in user_input.split(",")]
                        valid_departments = [dept for dept in departments if dept in system.departments or dept == "Tous"]
                        
                        # Mise à jour des accès
                        if system.set_department_access(person_id, valid_departments):
                            operation_feedback = f"Accès configurés pour {person_id}: {', '.join(valid_departments)}"
                        else:
                            operation_feedback = f"Erreur lors de la configuration des accès pour {person_id}"
                        feedback_timer = time.time()
                    else:
                        operation_feedback = f"Erreur: {person_id} n'existe pas dans la base de données"
                        feedback_timer = time.time()
                elif prompt_text == "Entrez le département à vérifier:":
                    department = user_input
                    if department in system.departments:
                        auth_department = department
                        if authenticated_person:
                            has_access = system.check_department_access(authenticated_person, department)
                            if has_access:
                                operation_feedback = f"{authenticated_person} a accès au département {department}"
                            else:
                                operation_feedback = f"{authenticated_person} n'a PAS accès au département {department}"
                        else:
                            operation_feedback = "Aucune personne authentifiée pour vérifier l'accès"
                        feedback_timer = time.time()
                    else:
                        operation_feedback = f"Erreur: Département {department} non reconnu"
                        feedback_timer = time.time()
                user_input = ""
            elif key == 27:  # Touche ESC
                input_active = False
                user_input = ""
            elif key == 8:  # Touche BACKSPACE
                user_input = user_input[:-1]
            elif 32 <= key <= 126:  # Caractères imprimables
                user_input += chr(key)
            continue  # Skip le reste des traitements de touche quand input_active
        
        # Traitement des commandes
        if key == ord('q'):
            break
        elif key == ord('r'):
            mode = "recognition"
            current_id = None
            authenticated_person = None  # Réinitialisation de l'authentification
            operation_feedback = "Mode reconnaissance activé"
            feedback_timer = time.time()
        elif key == ord('e'):
            mode = "registration"
            prompt_text = "Entrez l'ID de la nouvelle personne:"
            input_active = True
        elif key == ord('u'):
            prompt_text = "Entrez l'ID de la personne à mettre à jour:"
            input_active = True
        elif key == ord('i'):
            info = system.get_database_info()
            operation_feedback = f"Base de données: {info['total_persons']} personnes, {info['total_vectors']} vecteurs"
            feedback_timer = time.time()
        elif key == ord('d'):
            prompt_text = "Entrez l'ID de la personne à supprimer:"
            input_active = True
        elif key == ord('o'):
            removed = system.optimize_database()
            operation_feedback = f"Optimisation terminée: {removed} vecteurs dupliqués supprimés"
            feedback_timer = time.time()
        elif key == ord('a'):
            # Nouveau: Configuration des accès départementaux
            if mode == "recognition" and authenticated_person:
                # Vérification d'accès à un département
                prompt_text = "Entrez le département à vérifier:"
                input_active = True
            elif mode in ["registration", "update"] and current_id:
                # Configuration des accès pour une personne
                prompt_text = f"Départements (séparés par des virgules): {', '.join(system.departments)}"
                input_active = True
                if current_id in system.metadata:
                    user_input = ", ".join(system.metadata[current_id].get("department_access", []))
            else:
                prompt_text = "Entrez l'ID de la personne pour configurer l'accès:"
                input_active = True
        elif key == 32:  # ESPACE
            # Capture uniquement si un visage est détecté
            if len(faces) == 1 and mode in ["registration", "update"] and current_id:
                if mode == "registration":
                    # Ajout de métadonnées par défaut pour les nouveaux utilisateurs
                    default_metadata = {
                        "department_access": [], # Accès vide par défaut
                        "status": "authorized"  # Statut autorisé par défaut
                    }
                    
                    if system.register_face(frame, current_id, default_metadata):
                        operation_feedback = f"Visage enregistré pour {current_id}"
                        feedback_timer = time.time()
                    else:
                        operation_feedback = "Erreur: Échec de l'enregistrement"
                        feedback_timer = time.time()
                else:  # update
                    if system.update_face_vector(current_id, frame):
                        operation_feedback = f"Vecteur mis à jour pour {current_id}"
                        feedback_timer = time.time()
                    else:
                        operation_feedback = "Erreur: Échec de la mise à jour"
                        feedback_timer = time.time()
            
            # En mode reconnaissance, forcer la réauthentification
            elif mode == "recognition":
                authenticated_person = None
                welcome_displayed = False
    
    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Exécution de la démonstration
    demo_recognition_system()