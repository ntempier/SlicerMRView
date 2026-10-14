# MRView pour 3D Slicer

Module Python (scripted module) pour 3D Slicer 5.x qui reproduit les fonctions principales de **mrview** (MRtrix3) :

- **Tractogrammes `.tck`** en 3D et dans les coupes 2D, avec une coupe épaisse réglable (« crop to slab » de mrview).
- **Couleurs** : direction locale, direction des extrémités, couleur unie, aléatoire par fibre, longueur, scalaires `.tsf` (avec palette et plage réglables), opacité, épaisseur, rendu en tubes ombrés calculés par la carte graphique.
- **FOD / ODF** (`.mif`, `.mif.gz`, `.nii`, `.nii.gz`, convention SH de MRtrix3) en glyphes colorés par direction dans les coupes 2D et en 3D : taille, seuil, niveau de détail, normalisation, couleur unie. Le terme l=0 est chargé comme image de fond.
- **Filtrage en direct par ROI** : boîtes et sphères déplaçables à la souris, masques (labelmap, volume binaire, segments d'une segmentation). Rôles : *Inclure*, *Exclure*, *Extrémités*. Inclusions combinées en ET ou en OU.
- **Export** de la sélection en `.tck` (calculé sur le tractogramme complet) ou création d'un nouveau tractogramme.
- Images anatomiques `.mif` / NIfTI.

Aucune dépendance en plus de Slicer : les fichiers MRtrix et NIfTI sont lus avec numpy.

## Installation

1. Dans Slicer : **Edit → Application Settings → Modules → Additional module paths**.
2. Ajoute le dossier `SlicerMRView/MRView` (celui qui contient `MRView.py`).
3. Redémarre Slicer. Le module apparaît dans **Modules → Diffusion → MRView** (ou cherche « MRView » avec la loupe).

Pour en faire une extension installable, le dossier `SlicerMRView` contient déjà les `CMakeLists.txt` attendus par le Extension Wizard.

## Utilisation rapide

1. **Ouvrir des fichiers…** : sélectionne en une fois le T1, les FOD et le `.tck`. Le type est détecté automatiquement (une image 4D avec 15/28/45/66/91 volumes est traitée comme FOD). Les boutons *Tractogramme / FOD / Image* forcent le type.
2. Section **Tractographie** : couleur, épaisseur, tranche 2D, tubes.
3. Section **ODF / FOD** : taille et seuil des glyphes. *Aligner les coupes sur la grille* oriente les vues selon l'image FOD (utile pour une acquisition oblique).
4. Section **Filtrage par ROI** : *+ Boîte* / *+ Sphère* (vert = inclure, rouge = exclure), puis déplace les poignées dans les vues. Le rôle se change dans le tableau ; double-clic sur une ligne pour centrer les coupes sur la ROI.

### Gros tractogrammes

Le réglage **Fibres affichées (max)** (150 000 par défaut) limite l'affichage et le filtrage en direct à un sous-échantillon aléatoire. L'export `.tck` et « Nouveau tractogramme » utilisent toujours **toutes** les fibres.

## Détails techniques

- Coordonnées : les `.tck` sont en RAS scanner (mm), comme le monde de Slicer, donc aucun recalage n'est fait. Les transformations parentes de Slicer ne sont pas prises en compte dans cette version.
- SH : base orthonormée réelle de MRtrix3 (avec la phase de Condon-Shortley), directions en espace scanner. Vérifié sur les données de test : l'amplitude des FOD le long des fibres est maximale avec cette convention.
- Les glyphes ne sont calculés que pour les voxels de la coupe visible (et dans le champ de vue), avec réduction automatique du détail au-delà de 2,5 millions de sommets.
- Code testable hors Slicer : `python Testing/test_core.py` (numpy seul) depuis `SlicerMRView/MRView`.

## Fichiers

```
SlicerMRView/
  CMakeLists.txt              extension
  MRView/
    MRView.py                 module, interface, logique, test Slicer
    MRViewLib/
      mrtrix_io.py            lecture/écriture .tck .tsf .mif .nii
      sh.py                   harmoniques sphériques, maillage des glyphes
      tracts.py               opérations vectorisées sur les fibres, tests ROI
      layers.py               nœuds MRML : tractogrammes, FOD, ROI
      vtk_utils.py            conversions numpy ↔ VTK
    Testing/test_core.py      tests unitaires
    Resources/Icons/MRView.png
```

## Limites connues (v0.1)

- Pas de rendu des pics (`peaks`), ni de fichiers `.trk` / `.tck` multiples fusionnés.
- Les ROI masque d'une segmentation se mettent à jour quand la segmentation est modifiée, mais l'export de chaque segment peut être lent pendant l'édition.
- Le module n'a pas encore pu être lancé dans un vrai Slicer de mon côté (pas d'accès au téléchargement) : la logique a été testée sur tes données avec une maquette de l'API Slicer.
