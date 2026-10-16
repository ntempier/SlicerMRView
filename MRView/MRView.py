import os
import time

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)
from slicer.util import VTKObservationMixin

import importlib

from MRViewLib import mrtrix_io, sh, tracts, vtk_utils, bundle_fod, layers

# Slicer only re-executes MRView.py on "Reload" (or after a git pull without a full
# restart): reload the helper modules too, in dependency order, so they never go stale.
for _mod in (mrtrix_io, sh, tracts, vtk_utils, bundle_fod, layers):
    importlib.reload(_mod)
del _mod

from MRViewLib.layers import COLOR_MODES, DETAIL_LEVELS, ROI_MODE_COLORS


#
# Module
#

class MRView(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MRView"
        self.parent.categories = ["Diffusion"]
        self.parent.dependencies = []
        self.parent.contributors = ["Nicolas Tempier"]
        self.parent.helpText = """
Visualisation de tractographies et de FOD façon <b>mrview</b> (MRtrix3) :
<ul>
<li>tractogrammes <code>.tck</code> en 3D et dans les coupes 2D (coupe épaisse « slab »)</li>
<li>couleurs par direction, extrémités, couleur unie, aléatoire, longueur ou fichier <code>.tsf</code></li>
<li>glyphes FOD (harmoniques sphériques <code>.mif</code>/<code>.nii</code>) dans les coupes 2D et en 3D</li>
<li>filtrage des fibres en direct par ROI (boîte, sphère, masque ou segment) : inclure, exclure, extrémités</li>
</ul>
"""
        self.parent.acknowledgementText = "Inspiré de mrview (MRtrix3, www.mrtrix.org)."


#
# Widget
#

STYLE = """
QFrame#mrHeader {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16222a, stop:1 #3a6073);
  border-radius: 8px;
}
QLabel#mrTitle { color: white; font-size: 17px; font-weight: 700; }
QLabel#mrSubtitle { color: #d6e4ea; font-size: 11px; }
QPushButton#mrPrimary {
  background-color: #2f80ed; color: white; border: none; border-radius: 5px;
  padding: 7px 12px; font-weight: 600;
}
QPushButton#mrPrimary:hover { background-color: #4b93f2; }
QPushButton#mrPrimary:pressed { background-color: #1f6fd8; }
QPushButton#mrPrimary:disabled { background-color: #8aa9d6; }
QPushButton#mrSoft {
  border: 1px solid #2f80ed; color: #2f80ed; border-radius: 5px; padding: 5px 10px; background: transparent;
}
QPushButton#mrSoft:hover { background-color: rgba(47,128,237,0.12); }
QPushButton#mrInclude, QPushButton#mrExclude, QPushButton#mrEnds {
  border-radius: 5px; padding: 6px 8px; font-weight: 600; color: white; border: none;
}
QPushButton#mrInclude { background-color: #27ae60; }
QPushButton#mrInclude:hover { background-color: #2ecc71; }
QPushButton#mrExclude { background-color: #e74c3c; }
QPushButton#mrExclude:hover { background-color: #ef6a5c; }
QPushButton#mrEnds { background-color: #e0a800; }
QPushButton#mrEnds:hover { background-color: #f1c40f; }
QProgressBar#mrRatio {
  border: 1px solid rgba(128,128,128,0.6); border-radius: 5px; text-align: center; min-height: 18px;
}
QProgressBar#mrRatio::chunk { background-color: #27ae60; border-radius: 4px; }
QLabel#mrHint { color: gray; font-size: 11px; }
"""


def _icon(name, fallback_text=None):
    style = qt.QApplication.style()
    sp = getattr(qt.QStyle, name, None)
    if sp is None:
        return None
    return style.standardIcon(sp)


def _button(text, object_name=None, tooltip=None, icon=None):
    b = qt.QPushButton(text)
    if object_name:
        b.setObjectName(object_name)
    if tooltip:
        b.setToolTip(tooltip)
    if icon is not None:
        b.setIcon(icon)
    b.setCursor(qt.QCursor(qt.Qt.PointingHandCursor))
    return b


def _slider(minimum, maximum, step, value, decimals=2, suffix=""):
    s = ctk.ctkSliderWidget()
    s.minimum = minimum
    s.maximum = maximum
    s.singleStep = step
    s.pageStep = step * 10
    s.decimals = decimals
    s.value = value
    if suffix:
        s.suffix = suffix
    return s


def _hbox(*widgets, stretch_last=False):
    w = qt.QWidget()
    l = qt.QHBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 0)
    l.setSpacing(6)
    for x in widgets:
        if x is None:
            l.addStretch(1)
        else:
            l.addWidget(x)
    return w


def _qcolor(rgb):
    return qt.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _rgb(qcolor):
    return (qcolor.redF(), qcolor.greenF(), qcolor.blueF())


class MRViewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._syncing = False
        self._swatches = {}

    # ------------------------------------------------------------------ setup
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.parent.setStyleSheet(STYLE)
        self.logic = MRViewLogic()
        self.logic.add_listener(self._on_logic_event)

        self._build_header()
        self._build_data_section()
        self._build_bundle_section()
        self._build_fod_section()
        self._build_rgb_section()
        self._build_help_section()
        self.layout.addStretch(1)

        self.refresh_fod_list()
        self.refresh_tract_list()
        self.refresh_rgb_list()
        self.refresh_filter_stats()

    def cleanup(self):
        self.removeObservers()
        if self.logic is not None:
            self.logic.cleanup()

    # -- header -----------------------------------------------------------------
    def _build_header(self):
        frame = qt.QFrame()
        frame.setObjectName("mrHeader")
        l = qt.QVBoxLayout(frame)
        l.setContentsMargins(14, 10, 14, 10)
        l.setSpacing(2)
        title = qt.QLabel("MRView pour Slicer")
        title.setObjectName("mrTitle")
        sub = qt.QLabel("Tractographie · FOD · filtrage ROI en direct")
        sub.setObjectName("mrSubtitle")
        l.addWidget(title)
        l.addWidget(sub)
        self.layout.addWidget(frame)

    # -- data -------------------------------------------------------------------
    def _build_data_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Données"
        self.layout.addWidget(box)
        v = qt.QVBoxLayout(box)

        self.openButton = _button("  Ouvrir des fichiers…", "mrPrimary",
                                  "Ouvre .tck, .tsf, .mif, .mif.gz, .nii, .nii.gz.\n"
                                  "Le type (tractogramme, FOD, image) est détecté automatiquement.",
                                  _icon("SP_DialogOpenButton"))
        self.openButton.clicked.connect(self.on_open_files)
        v.addWidget(self.openButton)

        self.openTckButton = _button("Tractogramme", "mrSoft", "Ouvrir un ou plusieurs fichiers .tck")
        self.openFodButton = _button("FOD", "mrSoft", "Ouvrir une image d'harmoniques sphériques (FOD)")
        self.openImageButton = _button("Image", "mrSoft", "Ouvrir une image anatomique (T1, FA, …)")
        self.openRgbButton = _button("FA couleur", "mrSoft",
                                     "Ouvrir une carte de vecteurs à 3 volumes (tensor2metric -vector) en RVB, comme mrview")
        self.openTckButton.clicked.connect(lambda: self.on_open_files("tck"))
        self.openFodButton.clicked.connect(lambda: self.on_open_files("fod"))
        self.openImageButton.clicked.connect(lambda: self.on_open_files("image"))
        self.openRgbButton.clicked.connect(lambda: self.on_open_files("rgb"))
        v.addWidget(_hbox(self.openTckButton, self.openFodButton, self.openImageButton, self.openRgbButton))

        layout_label = qt.QLabel("Disposition :")
        buttons = []
        for text, layout_id, tip in (
                ("Classique", slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView, "3D en haut, 3 coupes en bas"),
                ("4 vues", slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView, "3 coupes + 3D"),
                ("3D", slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView, "Vue 3D seule"),
                ("Axiale", slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView, "Coupe axiale seule")):
            b = qt.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(False)
            b.clicked.connect(lambda checked=False, lid=layout_id: slicer.app.layoutManager().setLayout(lid))
            buttons.append(b)
        recenter = qt.QToolButton()
        recenter.setText("Recentrer")
        recenter.setToolTip("Recadrer les coupes et la vue 3D sur les données")
        recenter.clicked.connect(self.logic.recenter_views)
        v.addWidget(_hbox(layout_label, *buttons, None, recenter))

    # -- bundles (tractograms, their ROIs and display) ---------------------------
    def _build_bundle_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Faisceaux de fibres"
        self.layout.addWidget(box)
        self.tractBox = box
        v = qt.QVBoxLayout(box)

        hint = qt.QLabel("Chaque faisceau a ses propres ROI, sa couleur et ses réglages. "
                         "« Nouveau faisceau » repart des mêmes fibres pour isoler un autre faisceau.")
        hint.setObjectName("mrHint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.bundleTable = qt.QTableWidget()
        self.bundleTable.setColumnCount(5)
        self.bundleTable.setHorizontalHeaderLabels(["", "Faisceau", "Fibres", "Visible", ""])
        self.bundleTable.verticalHeader().setVisible(False)
        self.bundleTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.bundleTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        hh = self.bundleTable.horizontalHeader()
        hh.setSectionResizeMode(0, qt.QHeaderView.Fixed)
        hh.setSectionResizeMode(1, qt.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.bundleTable.setColumnWidth(0, 30)
        self.bundleTable.setMinimumHeight(110)
        self.bundleTable.setMaximumHeight(190)
        self.bundleTable.setToolTip("Clique sur un faisceau pour régler ses ROI et son affichage. "
                                    "Double-clic sur le nom pour le renommer.")
        self.bundleTable.connect("currentCellChanged(int,int,int,int)", self.on_bundle_row_changed)
        self.bundleTable.itemChanged.connect(self.on_bundle_item_changed)
        v.addWidget(self.bundleTable)

        self.newBundleButton = _button("+ Nouveau faisceau", "mrPrimary",
                                       "Nouveau faisceau à partir des mêmes fibres, sans ROI, avec sa propre couleur.\n"
                                       "Ajoute-lui ensuite des ROI pour isoler un autre faisceau.")
        self.dupBundleButton = _button("Dupliquer", "mrSoft",
                                       "Copie le faisceau sélectionné avec ses ROI, pour en faire une variante.")
        self.newBundleButton.clicked.connect(lambda: self.on_new_bundle(copy_rois=False))
        self.dupBundleButton.clicked.connect(lambda: self.on_new_bundle(copy_rois=True))
        v.addWidget(_hbox(self.newBundleButton, self.dupBundleButton))

        self.tractStats = qt.QLabel("Aucun tractogramme chargé.")
        self.tractStats.setObjectName("mrHint")
        self.tractStats.setWordWrap(True)
        self.tractStats.setTextFormat(qt.Qt.RichText)
        v.addWidget(self.tractStats)

        self.bundleTabs = qt.QTabWidget()
        self.bundleTabs.setObjectName("mrTabs")
        roi_page = qt.QWidget()
        self.roiPage = roi_page
        self._build_roi_page(qt.QVBoxLayout(roi_page))
        display_page = qt.QWidget()
        self._build_display_page(qt.QVBoxLayout(display_page))
        self.bundleTabs.addTab(roi_page, "ROI du faisceau")
        self.bundleTabs.addTab(display_page, "Affichage")
        v.addWidget(self.bundleTabs)

    def _build_display_page(self, v):
        self.tractForm = qt.QWidget()
        form = qt.QFormLayout(self.tractForm)
        form.setContentsMargins(0, 4, 0, 0)
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        v.addWidget(self.tractForm)
        v.addStretch(1)

        self.colorModeCombo = qt.QComboBox()
        for key, label in COLOR_MODES:
            self.colorModeCombo.addItem(label, key)
        form.addRow("Couleur", self.colorModeCombo)

        self.solidColorButton = ctk.ctkColorPickerButton()
        self.solidColorButton.displayColorName = False
        self.solidColorLabel = qt.QLabel("Couleur unie")
        form.addRow(self.solidColorLabel, self.solidColorButton)

        self.colormapCombo = slicer.qMRMLColorTableComboBox()
        self.colormapCombo.setMRMLScene(slicer.mrmlScene)
        self.colormapLabel = qt.QLabel("Palette")
        form.addRow(self.colormapLabel, self.colormapCombo)

        self.scalarRange = ctk.ctkRangeWidget()
        self.scalarRange.decimals = 2
        self.scalarRangeLabel = qt.QLabel("Plage")
        form.addRow(self.scalarRangeLabel, self.scalarRange)

        self.tsfButton = _button("Charger un fichier .tsf…", "mrSoft")
        self.tsfLabel = qt.QLabel("Scalaires")
        form.addRow(self.tsfLabel, self.tsfButton)

        self.tractOpacity = _slider(0.0, 1.0, 0.05, 1.0)
        form.addRow("Opacité", self.tractOpacity)
        self.tractWidth = _slider(1.0, 8.0, 0.5, 2.0, 1, " px")
        form.addRow("Épaisseur", self.tractWidth)
        self.tractStyle = qt.QComboBox()
        self.tractStyle.addItem("Lignes")
        self.tractStyle.addItem("Tubes (GPU)")
        self.tractStyle.setToolTip("Tubes : rendu ombré des lignes calculé par la carte graphique, sans coût de géométrie.")
        form.addRow("Rendu 3D", self.tractStyle)

        self.tractShow3D = qt.QCheckBox("Vue 3D")
        self.tractShow2D = qt.QCheckBox("Coupes 2D")
        self.tractShow3D.checked = True
        self.tractShow2D.checked = True
        form.addRow("Afficher dans", _hbox(self.tractShow3D, self.tractShow2D, None))

        self.slabSlider = _slider(0.5, 40.0, 0.5, 5.0, 1, " mm")
        self.slabSlider.setToolTip("Épaisseur de la tranche autour de chaque coupe 2D (équivalent du « crop to slab » de mrview).")
        form.addRow("Tranche 2D", self.slabSlider)

        self.maxDisplaySpin = qt.QSpinBox()
        self.maxDisplaySpin.setRange(1000, 20000000)
        self.maxDisplaySpin.setSingleStep(10000)
        self.maxDisplaySpin.setValue(150000)
        self.maxDisplaySpin.keyboardTracking = False
        self.maxDisplaySpin.setGroupSeparatorShown(True)
        self.maxDisplaySpin.setToolTip("Nombre maximum de fibres affichées et filtrées en direct.\n"
                                       "L'export .tck utilise toujours le tractogramme complet.")
        form.addRow("Fibres affichées (max)", self.maxDisplaySpin)

        self.colorModeCombo.currentIndexChanged.connect(self.on_color_mode)
        self.solidColorButton.connect("colorChanged(QColor)", self.on_solid_color)
        self.colormapCombo.connect("currentNodeChanged(vtkMRMLNode*)", self.on_colormap)
        self.scalarRange.connect("valuesChanged(double,double)", lambda a, b: self._set_tract("scalar_range", (a, b)))
        self.tsfButton.clicked.connect(self.on_load_tsf)
        self.tractOpacity.connect("valueChanged(double)", lambda x: self._set_tract("opacity", x))
        self.tractWidth.connect("valueChanged(double)", lambda x: self._set_tract("line_width", x))
        self.tractStyle.currentIndexChanged.connect(lambda i: self._set_tract("tubes", i == 1))
        self.tractShow3D.toggled.connect(lambda on: self._set_tract("show3d", on))
        self.tractShow2D.toggled.connect(lambda on: self._set_tract("show2d", on))
        self.slabSlider.connect("valueChanged(double)", lambda x: self._set_tract("slab_mm", x))
        self.maxDisplaySpin.valueChanged.connect(self.on_max_display)

        default_cmap = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeFileViridis.txt")
        if default_cmap is not None:
            self.colormapCombo.setCurrentNode(default_cmap)

    def _build_roi_page(self, v):
        hint = qt.QLabel("Ces ROI ne filtrent que le faisceau sélectionné. Ajoute une ROI puis déplace-la "
                         "avec ses poignées dans les vues : les fibres se mettent à jour en direct.")
        hint.setObjectName("mrHint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.addBoxInclude = _button("+ Boîte", "mrInclude", "Nouvelle ROI boîte (inclure)")
        self.addSphereInclude = _button("+ Sphère", "mrInclude", "Nouvelle ROI sphère (inclure)")
        self.addBoxExclude = _button("+ Boîte", "mrExclude", "Nouvelle ROI boîte (exclure)")
        self.addSphereExclude = _button("+ Sphère", "mrExclude", "Nouvelle ROI sphère (exclure)")
        grid = qt.QGridLayout()
        grid.addWidget(qt.QLabel("Inclure"), 0, 0)
        grid.addWidget(self.addBoxInclude, 0, 1)
        grid.addWidget(self.addSphereInclude, 0, 2)
        grid.addWidget(qt.QLabel("Exclure"), 1, 0)
        grid.addWidget(self.addBoxExclude, 1, 1)
        grid.addWidget(self.addSphereExclude, 1, 2)
        v.addLayout(grid)
        self.addBoxInclude.clicked.connect(lambda: self.on_add_roi("box", "include"))
        self.addSphereInclude.clicked.connect(lambda: self.on_add_roi("sphere", "include"))
        self.addBoxExclude.clicked.connect(lambda: self.on_add_roi("box", "exclude"))
        self.addSphereExclude.clicked.connect(lambda: self.on_add_roi("sphere", "exclude"))

        self.maskSelector = slicer.qMRMLNodeComboBox()
        self.maskSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"]
        self.maskSelector.noneEnabled = True
        self.maskSelector.addEnabled = False
        self.maskSelector.removeEnabled = False
        self.maskSelector.showChildNodeTypes = True
        self.maskSelector.setMRMLScene(slicer.mrmlScene)
        self.maskSelector.setToolTip("Segmentation (une ROI par segment), labelmap ou volume binaire")
        self.addMaskButton = _button("+ Masque", "mrSoft", "Ajouter ce masque / ces segments comme ROI de ce faisceau")
        self.addMaskButton.clicked.connect(self.on_add_mask)
        v.addWidget(_hbox(self.maskSelector, self.addMaskButton))

        self.roiTable = qt.QTableWidget()
        self.roiTable.setColumnCount(5)
        self.roiTable.setHorizontalHeaderLabels(["", "Nom", "Rôle", "Actif", ""])
        self.roiTable.verticalHeader().setVisible(False)
        self.roiTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.roiTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        hh = self.roiTable.horizontalHeader()
        hh.setSectionResizeMode(0, qt.QHeaderView.Fixed)
        hh.setSectionResizeMode(1, qt.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.roiTable.setColumnWidth(0, 14)
        self.roiTable.setMinimumHeight(110)
        self.roiTable.cellDoubleClicked.connect(lambda r, c: self.logic.jump_to_roi(r))
        self.roiTable.itemChanged.connect(self.on_roi_item_changed)
        v.addWidget(self.roiTable)

        self.includeAll = qt.QRadioButton("toutes (ET)")
        self.includeAny = qt.QRadioButton("au moins une (OU)")
        self.includeAll.checked = True
        self.includeAll.toggled.connect(self.on_include_mode)
        v.addWidget(_hbox(qt.QLabel("Inclusions requises :"), self.includeAll, self.includeAny, None))

        self.liveCheck = qt.QCheckBox("Filtrage en direct")
        self.liveCheck.checked = True
        self.applyButton = _button("Appliquer", "mrSoft")
        self.applyButton.enabled = False
        self.showRois = qt.QCheckBox("Afficher les ROI")
        self.showRois.checked = True
        self.showRois.setToolTip("Seules les ROI du faisceau sélectionné sont affichées dans les vues.")
        self.liveCheck.toggled.connect(self.on_live_toggled)
        self.applyButton.clicked.connect(self.logic.apply_filters)
        self.showRois.toggled.connect(self.logic.set_rois_visible)
        v.addWidget(_hbox(self.liveCheck, self.applyButton, None, self.showRois))

        self.ratioBar = qt.QProgressBar()
        self.ratioBar.setObjectName("mrRatio")
        self.ratioBar.setRange(0, 1000)
        self.ratioBar.setValue(1000)
        self.ratioBar.setFormat("—")
        v.addWidget(self.ratioBar)
        self.filterStats = qt.QLabel("")
        self.filterStats.setObjectName("mrHint")
        self.filterStats.setWordWrap(True)
        self.filterStats.setTextFormat(qt.Qt.RichText)
        v.addWidget(self.filterStats)

        self.bundleFodButton = _button("Voir les FOD de ce faisceau", "mrSoft",
                                       "N'affiche que les FOD des voxels traversés par les fibres sélectionnées de ce faisceau "
                                       "(réglages dans la section ODF / FOD).")
        self.bundleFodButton.clicked.connect(self.on_show_bundle_fod)
        v.addWidget(self.bundleFodButton)

        self.exportButton = _button("Enregistrer la sélection (.tck)…", "mrPrimary",
                                    "Applique les ROI de ce faisceau au tractogramme complet et enregistre le résultat",
                                    _icon("SP_DialogSaveButton"))
        self.toLayerButton = _button("Figer en tractogramme", "mrSoft",
                                     "Crée un nouveau tractogramme indépendant à partir des fibres sélectionnées")
        self.clearRoisButton = _button("Supprimer les ROI", "mrSoft", "Supprime toutes les ROI de ce faisceau")
        self.exportButton.clicked.connect(self.on_export_selection)
        self.toLayerButton.clicked.connect(self.on_selection_to_layer)
        self.clearRoisButton.clicked.connect(self.on_clear_rois)
        v.addWidget(self.exportButton)
        v.addWidget(_hbox(self.toLayerButton, self.clearRoisButton))

    # -- FOD --------------------------------------------------------------------
    def _build_fod_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "ODF / FOD"
        self.fodBox = box
        self.layout.addWidget(box)
        v = qt.QVBoxLayout(box)

        self.fodCombo = qt.QComboBox()
        self.fodCombo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.fodVisible = qt.QCheckBox("Visible")
        self.fodVisible.checked = True
        self.fodRemove = qt.QToolButton()
        trash = _icon("SP_TrashIcon")
        if trash is not None:
            self.fodRemove.setIcon(trash)
        else:
            self.fodRemove.setText("✕")
        self.fodRemove.setToolTip("Fermer cette image FOD")
        v.addWidget(_hbox(self.fodCombo, self.fodVisible, self.fodRemove))

        self.fodInfo = qt.QLabel("Aucune image FOD chargée.")
        self.fodInfo.setObjectName("mrHint")
        self.fodInfo.setWordWrap(True)
        v.addWidget(self.fodInfo)

        self.fodForm = qt.QWidget()
        form = qt.QFormLayout(self.fodForm)
        form.setContentsMargins(0, 4, 0, 0)
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        v.addWidget(self.fodForm)

        self.fodShow2D = qt.QCheckBox("Coupes 2D")
        self.fodShow3D = qt.QCheckBox("Vue 3D")
        self.fodShow2D.checked = True
        self.fodShow3D.setToolTip("Affiche en 3D les glyphes des coupes visibles")
        form.addRow("Afficher dans", _hbox(self.fodShow2D, self.fodShow3D, None))

        self.fodScale = _slider(0.1, 5.0, 0.05, 1.0)
        form.addRow("Taille", self.fodScale)
        self.fodThreshold = _slider(0.0, 1.0, 0.01, 0.1)
        self.fodThreshold.setToolTip("Masque les glyphes dont l'amplitude maximale est inférieure à cette fraction "
                                     "de l'amplitude de référence (95e percentile).")
        form.addRow("Seuil d'amplitude", self.fodThreshold)
        self.fodDetail = qt.QComboBox()
        for level, label in DETAIL_LEVELS:
            self.fodDetail.addItem(label, level)
        self.fodDetail.setCurrentIndex(1)
        form.addRow("Détail", self.fodDetail)
        self.fodNormalize = qt.QCheckBox("Normaliser chaque glyphe")
        form.addRow("", self.fodNormalize)
        self.fodColorMode = qt.QComboBox()
        self.fodColorMode.addItem("Direction", "direction")
        self.fodColorMode.addItem("Couleur unie", "solid")
        self.fodColor = ctk.ctkColorPickerButton()
        self.fodColor.displayColorName = False
        self.fodColor.color = _qcolor((0.85, 0.85, 0.85))
        form.addRow("Couleur", _hbox(self.fodColorMode, self.fodColor))
        self.fodOpacity = _slider(0.0, 1.0, 0.05, 1.0)
        form.addRow("Opacité", self.fodOpacity)

        # -- FOD of a bundle -----------------------------------------------------
        self.fodBundleBox = qt.QGroupBox("FOD d'un faisceau")
        self.fodBundleBox.setToolTip("Ne montre que les FOD utilisées par un faisceau : celles des voxels "
                                     "traversés par ses fibres sélectionnées (mises à jour quand ses ROI bougent).")
        bform = qt.QFormLayout(self.fodBundleBox)
        bform.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        self.fodBundleCombo = qt.QComboBox()
        self.fodBundleCombo.setToolTip("Faisceau dont on veut voir les FOD")
        bform.addRow("Faisceau", self.fodBundleCombo)
        self.fodBundleMin = qt.QSpinBox()
        self.fodBundleMin.setRange(1, 100000)
        self.fodBundleMin.setValue(1)
        self.fodBundleMin.setSuffix(" fibre(s)")
        self.fodBundleMin.setToolTip("Nombre minimum de fibres du faisceau qui doivent traverser un voxel pour afficher sa FOD.")
        bform.addRow("Min. par voxel", self.fodBundleMin)
        self.fodBundleLobes = qt.QCheckBox("Seulement les lobes alignés sur les fibres")
        self.fodBundleLobes.setToolTip("Dans chaque voxel, ne garde que les lobes de la FOD orientés comme les fibres "
                                       "du faisceau qui le traversent.")
        bform.addRow("", self.fodBundleLobes)
        self.fodBundleAngle = _slider(5.0, 60.0, 1.0, 25.0, 0, "°")
        self.fodBundleAngle.setToolTip("Écart angulaire toléré entre un lobe et la direction des fibres.")
        self.fodBundleAngleLabel = qt.QLabel("Tolérance")
        bform.addRow(self.fodBundleAngleLabel, self.fodBundleAngle)
        self.fodBundleInfo = qt.QLabel("")
        self.fodBundleInfo.setObjectName("mrHint")
        self.fodBundleInfo.setWordWrap(True)
        bform.addRow(self.fodBundleInfo)
        v.addWidget(self.fodBundleBox)

        self.fodBundleCombo.currentIndexChanged.connect(self.on_fod_bundle)
        self.fodBundleMin.valueChanged.connect(lambda x: self._set_fod("bundle_min", int(x)))
        self.fodBundleLobes.toggled.connect(self.on_fod_bundle_lobes)
        self.fodBundleAngle.connect("valueChanged(double)", lambda x: self._set_fod("bundle_angle", x))

        self.alignButton = _button("Aligner les coupes sur la grille", "mrSoft",
                                   "Oriente les coupes 2D selon les axes de l'image FOD (utile si l'acquisition est oblique)")
        self.l0Button = _button("Image l=0 en fond", "mrSoft", "Utilise le terme l=0 des FOD comme image de fond")
        v.addWidget(_hbox(self.alignButton, self.l0Button))

        self.fodCount = qt.QLabel("")
        self.fodCount.setObjectName("mrHint")
        v.addWidget(self.fodCount)

        self.fodCombo.currentIndexChanged.connect(self.on_fod_selected)
        self.fodVisible.toggled.connect(lambda on: self._set_fod("visible", on))
        self.fodRemove.clicked.connect(self.on_remove_fod)
        self.fodShow2D.toggled.connect(lambda on: self._set_fod("show2d", on))
        self.fodShow3D.toggled.connect(lambda on: self._set_fod("show3d", on))
        self.fodScale.connect("valueChanged(double)", lambda x: self._set_fod("scale", x))
        self.fodThreshold.connect("valueChanged(double)", lambda x: self._set_fod("threshold", x))
        self.fodDetail.currentIndexChanged.connect(lambda i: self._set_fod("level", self.fodDetail.itemData(i)))
        self.fodNormalize.toggled.connect(lambda on: self._set_fod("normalize", on))
        self.fodColorMode.currentIndexChanged.connect(lambda i: self._set_fod("color_mode", self.fodColorMode.itemData(i)))
        self.fodColor.connect("colorChanged(QColor)", lambda c: self._set_fod("solid_color", _rgb(c)))
        self.fodOpacity.connect("valueChanged(double)", lambda x: self._set_fod("opacity", x))
        self.alignButton.clicked.connect(self.on_align_to_fod)
        self.l0Button.clicked.connect(self.on_l0_background)

    # -- colour-coded FA / vector maps --------------------------------------------
    def _build_rgb_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "FA couleur (RVB)"
        self.layout.addWidget(box)
        v = qt.QVBoxLayout(box)

        self.rgbCombo = qt.QComboBox()
        self.rgbCombo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.rgbRemove = qt.QToolButton()
        trash = _icon("SP_TrashIcon")
        if trash is not None:
            self.rgbRemove.setIcon(trash)
        else:
            self.rgbRemove.setText("✕")
        self.rgbRemove.setToolTip("Fermer cette carte")
        self.rgbVisible = qt.QCheckBox("Visible")
        self.rgbVisible.checked = True
        v.addWidget(_hbox(self.rgbCombo, self.rgbVisible, self.rgbRemove))

        self.rgbInfo = qt.QLabel("")
        self.rgbInfo.setObjectName("mrHint")
        self.rgbInfo.setWordWrap(True)
        v.addWidget(self.rgbInfo)

        self.rgbForm = qt.QWidget()
        form = qt.QFormLayout(self.rgbForm)
        form.setContentsMargins(0, 4, 0, 0)
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        v.addWidget(self.rgbForm)

        self.rgbGain = _slider(0.1, 5.0, 0.05, 1.0)
        self.rgbGain.setToolTip("Multiplie l'intensité des couleurs (1 = le 99,5e percentile est à pleine luminosité).")
        form.addRow("Luminosité", self.rgbGain)
        self.rgbMode = qt.QComboBox()
        self.rgbMode.addItem("Image de fond", "background")
        self.rgbMode.addItem("Superposée à l'image de fond", "overlay")
        self.rgbMode.setToolTip("Superposée : la carte RVB s'affiche par-dessus l'image qui était en fond (T1, …).")
        form.addRow("Affichage", self.rgbMode)
        self.rgbOpacity = _slider(0.0, 1.0, 0.05, 0.6)
        self.rgbOpacityLabel = qt.QLabel("Opacité")
        form.addRow(self.rgbOpacityLabel, self.rgbOpacity)
        self.rgbInterpolate = qt.QCheckBox("Interpolation")
        self.rgbInterpolate.checked = True
        self.rgbInterpolate.setToolTip("Décocher pour voir les voxels nets, comme dans mrview.")
        form.addRow("", self.rgbInterpolate)
        self.rgbAlign = _button("Aligner les coupes sur la grille", "mrSoft",
                                "Oriente les coupes 2D selon les axes de cette carte (utile si l'acquisition est oblique)")
        form.addRow("", self.rgbAlign)

        self.rgbEmpty = qt.QLabel("Aucune carte chargée. Bouton « FA couleur » (ou « Ouvrir des fichiers… ») "
                                  "pour une image à 3 volumes, par ex. la sortie de tensor2metric -vector.")
        self.rgbEmpty.setObjectName("mrHint")
        self.rgbEmpty.setWordWrap(True)
        v.addWidget(self.rgbEmpty)

        self._rgb_gain_timer = qt.QTimer()
        self._rgb_gain_timer.setSingleShot(True)
        self._rgb_gain_timer.setInterval(30)
        self._rgb_gain_timer.connect("timeout()", self._apply_rgb_gain)

        self.rgbCombo.currentIndexChanged.connect(self.on_rgb_selected)
        self.rgbRemove.clicked.connect(self.on_remove_rgb)
        self.rgbVisible.toggled.connect(self.on_rgb_visible)
        self.rgbInterpolate.toggled.connect(self.on_rgb_interpolate)
        self.rgbAlign.clicked.connect(self.on_rgb_align)
        self.rgbGain.connect("valueChanged(double)", lambda x: self._syncing or self._rgb_gain_timer.start())
        self.rgbMode.currentIndexChanged.connect(self.on_rgb_mode)
        self.rgbOpacity.connect("valueChanged(double)", self.on_rgb_opacity)

    def current_rgb(self):
        i = self.rgbCombo.currentIndex
        if 0 <= i < len(self.logic.rgb_layers):
            return self.logic.rgb_layers[i]
        return None

    def refresh_rgb_list(self, select=None):
        self._syncing = True
        try:
            current = select if select is not None else self.current_rgb()
            self.rgbCombo.clear()
            for layer in self.logic.rgb_layers:
                self.rgbCombo.addItem(layer.name)
            if current in self.logic.rgb_layers:
                self.rgbCombo.setCurrentIndex(self.logic.rgb_layers.index(current))
            elif self.logic.rgb_layers:
                self.rgbCombo.setCurrentIndex(len(self.logic.rgb_layers) - 1)
        finally:
            self._syncing = False
        self.sync_rgb_controls()

    def sync_rgb_controls(self):
        layer = self.current_rgb()
        has = layer is not None
        self.rgbCombo.visible = has
        self.rgbRemove.visible = has
        self.rgbVisible.visible = has
        self.rgbForm.visible = has
        self.rgbInfo.visible = has
        self.rgbEmpty.visible = not has
        if not has:
            return
        self._syncing = True
        try:
            self.rgbInfo.text = layer.info_text()
            self.rgbGain.value = layer.gain
            self.rgbMode.setCurrentIndex(1 if layer.mode == "overlay" else 0)
            self.rgbOpacity.value = layer.opacity
            self.rgbVisible.checked = layer.visible
            dn = layer.node.GetDisplayNode() if layer.node is not None else None
            self.rgbInterpolate.checked = bool(dn.GetInterpolate()) if dn is not None else True
            self.rgbOpacity.enabled = layer.mode == "overlay"
            self.rgbOpacityLabel.enabled = layer.mode == "overlay"
        finally:
            self._syncing = False

    def on_rgb_selected(self, index):
        if self._syncing:
            return
        self.sync_rgb_controls()
        layer = self.current_rgb()
        if layer is not None:
            self.logic.show_rgb(layer)

    def _apply_rgb_gain(self):
        layer = self.current_rgb()
        if layer is not None:
            layer.set_gain(self.rgbGain.value)

    def on_rgb_mode(self, index):
        if self._syncing:
            return
        layer = self.current_rgb()
        if layer is None:
            return
        layer.mode = self.rgbMode.itemData(index)
        self.logic.show_rgb(layer)
        self.sync_rgb_controls()

    def on_rgb_opacity(self, value):
        if self._syncing:
            return
        layer = self.current_rgb()
        if layer is None:
            return
        layer.opacity = value
        if layer.mode == "overlay":
            self.logic.show_rgb(layer)

    def on_rgb_visible(self, on):
        layer = self.current_rgb()
        if self._syncing or layer is None:
            return
        layer.visible = on
        self.logic.show_rgb(layer)

    def on_rgb_interpolate(self, on):
        layer = self.current_rgb()
        if self._syncing or layer is None or layer.node is None:
            return
        dn = layer.node.GetDisplayNode()
        if dn is not None:
            dn.SetInterpolate(1 if on else 0)

    def on_rgb_align(self):
        layer = self.current_rgb()
        if layer is not None and layer.node is not None:
            for sn in layers.slice_nodes():
                sn.RotateToVolumePlane(layer.node)
            self.logic.request_views()

    def on_remove_rgb(self):
        layer = self.current_rgb()
        if layer is not None:
            self.logic.remove_rgb(layer)

    def _build_help_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Astuces"
        box.collapsed = True
        self.layout.addWidget(box)
        v = qt.QVBoxLayout(box)
        tips = qt.QLabel(
            "<ul style='margin-left:-20px'>"
            "<li><b>Défilement des coupes</b> : molette ou flèches ↑/↓ dans une vue 2D.</li>"
            "<li><b>Déplacer le curseur 3D</b> : Maj + déplacement de la souris.</li>"
            "<li><b>Plusieurs faisceaux</b> : charge un tractogramme, puis « + Nouveau faisceau » pour chaque faisceau "
            "à isoler. Chacun a ses ROI (seules celles du faisceau sélectionné sont affichées) et sa couleur "
            "(clic sur la pastille de couleur).</li>"
            "<li><b>FOD d'un faisceau</b> : « Voir les FOD de ce faisceau » (onglet ROI) ne montre que les FOD des voxels "
            "traversés par ses fibres ; option pour ne garder que les lobes alignés sur les fibres.</li>"
            "<li><b>ROI</b> : double-clic sur une ligne du tableau pour centrer les coupes dessus.</li>"
            "<li><b>Gros tractogrammes</b> : réduis « Fibres affichées (max) » pour garder un filtrage fluide ; "
            "l'export utilise toujours toutes les fibres.</li>"
            "<li><b>Extrémités</b> : la fibre doit commencer ou finir dans la ROI (connectivité).</li>"
            "<li><b>Tubes (GPU)</b> donne un rendu ombré sans ralentir l'affichage.</li>"
            "</ul>")
        tips.setWordWrap(True)
        tips.setTextFormat(qt.Qt.RichText)
        v.addWidget(tips)

    # ------------------------------------------------------------ data actions
    def on_open_files(self, kind=None):
        filters = {
            None: "Fichiers MRtrix / NIfTI (*.tck *.tsf *.mif *.mif.gz *.mih *.nii *.nii.gz);;Tous (*)",
            "tck": "Tractogrammes (*.tck);;Tous (*)",
            "fod": "Images FOD (*.mif *.mif.gz *.mih *.nii *.nii.gz);;Tous (*)",
            "image": "Images (*.mif *.mif.gz *.mih *.nii *.nii.gz *.nrrd);;Tous (*)",
            "rgb": "Cartes de vecteurs (*.mif *.mif.gz *.mih *.nii *.nii.gz);;Tous (*)",
        }
        start = self._last_dir()
        paths = qt.QFileDialog.getOpenFileNames(slicer.util.mainWindow(), "Ouvrir", start, filters[kind])
        if not paths:
            return
        self._remember_dir(paths[0])
        # load images first so tractograms appear on top of a sensible background
        def order(p):
            low = p.lower()
            return 2 if low.endswith(".tck") else (3 if low.endswith(".tsf") else 1)
        for path in sorted(paths, key=order):
            self._load_one(path, kind)

    def _load_one(self, path, kind=None):
        low = path.lower()
        if kind is None:
            if low.endswith(".tck"):
                kind = "tck"
            elif low.endswith(".tsf"):
                kind = "tsf"
            elif mrtrix_io.looks_like_fod(path):
                kind = "fod"
            elif mrtrix_io.looks_like_vector(path):
                kind = "rgb"
            else:
                kind = "image"
        name = os.path.basename(path)
        try:
            with slicer.util.tryWithErrorDisplay("Impossible d'ouvrir %s" % name, waitCursor=True):
                slicer.util.showStatusMessage("Chargement de %s…" % name)
                slicer.app.processEvents()
                t0 = time.time()
                if kind == "tck":
                    self.logic.load_tractogram(path)
                elif kind == "tsf":
                    layer = self.current_tract()
                    if layer is None:
                        raise ValueError("Charge d'abord un tractogramme avant son fichier .tsf.")
                    self.logic.load_tsf(layer, path)
                    self._select_color_mode("tsf")
                elif kind == "fod":
                    self.logic.load_fod(path)
                elif kind == "rgb":
                    self.logic.load_rgb(path)
                else:
                    self.logic.load_image(path)
                slicer.util.showStatusMessage("%s chargé en %.1f s" % (name, time.time() - t0), 4000)
        except Exception:
            pass

    def _last_dir(self):
        return qt.QSettings().value("MRView/lastDir", "") or ""

    def _remember_dir(self, path):
        qt.QSettings().setValue("MRView/lastDir", os.path.dirname(path))

    # --------------------------------------------------------- bundle actions
    def current_tract(self):
        i = self.bundleTable.currentRow()
        if 0 <= i < len(self.logic.tract_layers):
            return self.logic.tract_layers[i]
        return None

    def refresh_tract_list(self, select=None):
        """Rebuild the bundle table (call when bundles are added, removed or recoloured)."""
        self._syncing = True
        try:
            current = select if select is not None else self.current_tract()
            bundles = self.logic.tract_layers
            self._swatches = {}
            self.bundleTable.setRowCount(len(bundles))
            for row, layer in enumerate(bundles):
                swatch = qt.QToolButton()
                swatch.setAutoRaise(False)
                swatch.setFixedSize(22, 18)
                swatch.setCursor(qt.QCursor(qt.Qt.PointingHandCursor))
                swatch.setStyleSheet(_swatch_style(layer))
                swatch.setToolTip("Couleur du faisceau : clic pour choisir une couleur unie")
                swatch.clicked.connect(lambda checked=False, l=layer: self.on_pick_bundle_color(l))
                self._swatches[layer.uid] = swatch
                self.bundleTable.setCellWidget(row, 0, _centered(swatch))

                name = qt.QTableWidgetItem(layer.name)
                name.setToolTip(layer.path or layer.name)
                name.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEditable)
                self.bundleTable.setItem(row, 1, name)

                count = qt.QTableWidgetItem(_count_text(layer))
                count.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                count.setTextAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
                self.bundleTable.setItem(row, 2, count)

                visible = qt.QCheckBox()
                visible.checked = layer.visible
                visible.setToolTip("Afficher ce faisceau")
                visible.toggled.connect(lambda on, l=layer: self.on_bundle_visible(l, on))
                self.bundleTable.setCellWidget(row, 3, _centered(visible))

                delete = qt.QToolButton()
                trash = _icon("SP_TrashIcon")
                if trash is not None:
                    delete.setIcon(trash)
                else:
                    delete.setText("✕")
                delete.setAutoRaise(True)
                delete.setToolTip("Fermer ce faisceau (et ses ROI)")
                delete.clicked.connect(lambda checked=False, l=layer: self.on_remove_tract(l))
                self.bundleTable.setCellWidget(row, 4, _centered(delete))
            if current in bundles:
                self.bundleTable.setCurrentCell(bundles.index(current), 1)
            elif bundles:
                self.bundleTable.setCurrentCell(len(bundles) - 1, 1)
        finally:
            self._syncing = False
        self.logic.set_active_bundle(self.current_tract())
        self.sync_tract_controls()
        self.refresh_roi_table()
        self.refresh_fod_bundle_combo()

    def refresh_bundle_counts(self):
        self._syncing = True
        try:
            for row, layer in enumerate(self.logic.tract_layers):
                item = self.bundleTable.item(row, 2)
                if item is not None:
                    item.setText(_count_text(layer))
        finally:
            self._syncing = False

    def refresh_bundle_swatch(self, layer):
        swatch = self._swatches.get(layer.uid) if layer is not None else None
        if swatch is not None:
            swatch.setStyleSheet(_swatch_style(layer))

    def on_bundle_row_changed(self, row, col, prev_row, prev_col):
        if self._syncing or row == prev_row:
            return
        self.logic.set_active_bundle(self.current_tract())
        self.sync_tract_controls()
        self.refresh_roi_table()
        self.refresh_filter_stats()

    def on_bundle_item_changed(self, item):
        if self._syncing or item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self.logic.tract_layers):
            name = item.text().strip()
            if name:
                self.logic.tract_layers[row].rename(name)
                self.refresh_fod_bundle_combo()
                self.sync_fod_controls()

    def on_bundle_visible(self, layer, on):
        layer.visible = on
        layer.apply_display()
        layer.invalidate_slabs()
        self.logic.request_views()

    def on_pick_bundle_color(self, layer):
        color = qt.QColorDialog.getColor(_qcolor(layer.solid_color), slicer.util.mainWindow(),
                                         "Couleur de « %s »" % layer.name)
        if not color.isValid():
            return
        layer.solid_color = _rgb(color)
        layer.color_mode = "solid"
        layer.apply_display()
        self.refresh_bundle_swatch(layer)
        if layer is self.current_tract():
            self.sync_tract_controls()

    def on_new_bundle(self, copy_rois=False):
        source = self.current_tract()
        if source is None:
            slicer.util.warningDisplay("Charge d'abord un tractogramme (bouton « Tractogramme »).")
            return
        with slicer.util.tryWithErrorDisplay("Impossible de créer le faisceau", waitCursor=True):
            self.logic.new_bundle(source, copy_rois=copy_rois)
            self.bundleTabs.setCurrentIndex(0)

    def sync_tract_controls(self):
        layer = self.current_tract()
        has = layer is not None
        for w in (self.tractForm, self.roiPage, self.dupBundleButton, self.newBundleButton):
            if w is not None:
                w.enabled = has
        if not has:
            self.tractStats.text = "Aucun tractogramme chargé. Utilise « Ouvrir des fichiers… »."
            self._update_color_rows("direction")
            return
        self._syncing = True
        try:
            self.tractStats.text = layer.stats_text()
            self.colorModeCombo.setCurrentIndex([k for k, _ in COLOR_MODES].index(layer.color_mode))
            self.solidColorButton.color = _qcolor(layer.solid_color)
            if layer.colormap_id:
                self.colormapCombo.setCurrentNodeID(layer.colormap_id)
            self.tractOpacity.value = layer.opacity
            self.tractWidth.value = layer.line_width
            self.tractStyle.setCurrentIndex(1 if layer.tubes else 0)
            self.tractShow3D.checked = layer.show3d
            self.tractShow2D.checked = layer.show2d
            self.slabSlider.value = layer.slab_mm
            self.maxDisplaySpin.setValue(layer.max_display)
            self.includeAll.checked = layer.include_mode == "all"
            self.includeAny.checked = layer.include_mode == "any"
            self._sync_scalar_range(layer)
            self._update_color_rows(layer.color_mode)
        finally:
            self._syncing = False

    def _sync_scalar_range(self, layer):
        if layer.color_mode not in ("length", "tsf"):
            return
        values = layer.scalar_values(layer.color_mode)
        if values is None or values.size == 0:
            return
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        if layer.scalar_range is None:
            layer.scalar_range = layer.default_scalar_range(layer.color_mode)
        self.scalarRange.setRange(lo, hi)
        self.scalarRange.singleStep = max((hi - lo) / 200.0, 1e-4)
        self.scalarRange.setValues(layer.scalar_range[0], layer.scalar_range[1])

    def _update_color_rows(self, mode):
        solid = mode == "solid"
        scalar = mode in ("length", "tsf")
        for w in (self.solidColorButton, self.solidColorLabel):
            w.setVisible(solid)
        for w in (self.colormapCombo, self.colormapLabel, self.scalarRange, self.scalarRangeLabel):
            w.setVisible(scalar)
        for w in (self.tsfButton, self.tsfLabel):
            w.setVisible(mode == "tsf")

    def _set_tract(self, attr, value):
        if self._syncing:
            return
        layer = self.current_tract()
        if layer is None:
            return
        setattr(layer, attr, value)
        if attr == "slab_mm":
            layer.invalidate_slabs()
            self.logic.request_views()
        elif attr in ("visible", "show2d"):
            layer.apply_display()
            layer.invalidate_slabs()
            self.logic.request_views()
        else:
            layer.apply_display()

    def on_solid_color(self, color):
        if self._syncing:
            return
        self._set_tract("solid_color", _rgb(color))
        self.refresh_bundle_swatch(self.current_tract())

    def _select_color_mode(self, mode):
        keys = [k for k, _ in COLOR_MODES]
        self.colorModeCombo.setCurrentIndex(keys.index(mode))

    def on_color_mode(self, index):
        mode = self.colorModeCombo.itemData(index)
        self._update_color_rows(mode)
        if self._syncing:
            return
        layer = self.current_tract()
        if layer is None:
            return
        if mode == "tsf" and layer.full_tsf is None:
            self.on_load_tsf()
            if layer.full_tsf is None:
                self._syncing = True
                self.colorModeCombo.setCurrentIndex([k for k, _ in COLOR_MODES].index(layer.color_mode))
                self._syncing = False
                self._update_color_rows(layer.color_mode)
                return
        if mode != layer.color_mode:
            layer.scalar_range = None
        layer.color_mode = mode
        if mode in ("length", "tsf") and not layer.colormap_id:
            layer.colormap_id = self.colormapCombo.currentNodeID
        with _WaitCursor():
            layer.apply_display()
        self.refresh_bundle_swatch(layer)
        self._syncing = True
        self._sync_scalar_range(layer)
        self._syncing = False

    def on_colormap(self, node):
        if self._syncing:
            return
        layer = self.current_tract()
        if layer is not None and node is not None:
            layer.colormap_id = node.GetID()
            layer.apply_display()

    def on_load_tsf(self):
        layer = self.current_tract()
        if layer is None:
            return
        path = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Fichier de scalaires", self._last_dir(),
                                              "Track scalar files (*.tsf);;Tous (*)")
        if not path:
            return
        self._remember_dir(path)
        with slicer.util.tryWithErrorDisplay("Impossible de charger le fichier .tsf", waitCursor=True):
            self.logic.load_tsf(layer, path)
            self._select_color_mode("tsf")
            layer.color_mode = "tsf"
            if not layer.colormap_id:
                layer.colormap_id = self.colormapCombo.currentNodeID
            layer.apply_display()
            self.sync_tract_controls()
            self.refresh_bundle_swatch(layer)

    def on_max_display(self, value):
        if self._syncing:
            return
        layer = self.current_tract()
        if layer is None:
            return
        with _WaitCursor():
            self.logic.set_max_display(layer, int(value))
        self.sync_tract_controls()

    def on_remove_tract(self, layer=None):
        layer = layer or self.current_tract()
        if layer is None:
            return
        if layer.rois and not slicer.util.confirmYesNoDisplay(
                "Fermer « %s » et supprimer ses %d ROI ?" % (layer.name, len(layer.rois))):
            return
        self.logic.remove_tract(layer)

    # ------------------------------------------------------------ FOD actions
    def current_fod(self):
        i = self.fodCombo.currentIndex
        if 0 <= i < len(self.logic.fod_layers):
            return self.logic.fod_layers[i]
        return None

    def refresh_fod_list(self, select=None):
        self._syncing = True
        try:
            current = select if select is not None else self.current_fod()
            self.fodCombo.clear()
            for layer in self.logic.fod_layers:
                self.fodCombo.addItem(layer.name)
            if current in self.logic.fod_layers:
                self.fodCombo.setCurrentIndex(self.logic.fod_layers.index(current))
            elif self.logic.fod_layers:
                self.fodCombo.setCurrentIndex(len(self.logic.fod_layers) - 1)
        finally:
            self._syncing = False
        self.sync_fod_controls()

    def sync_fod_controls(self):
        layer = self.current_fod()
        has = layer is not None
        for w in (self.fodForm, self.fodVisible, self.fodRemove, self.alignButton, self.l0Button, self.fodBundleBox):
            w.enabled = has
        if not has:
            self.fodInfo.text = "Aucune image FOD chargée."
            self.fodCount.text = ""
            self.refresh_fod_bundle_combo()
            return
        self._syncing = True
        try:
            self.fodInfo.text = layer.info_text()
            self.fodVisible.checked = layer.visible
            self.fodShow2D.checked = layer.show2d
            self.fodShow3D.checked = layer.show3d
            self.fodScale.value = layer.scale
            self.fodThreshold.value = layer.threshold
            self.fodDetail.setCurrentIndex([l for l, _ in DETAIL_LEVELS].index(layer.level))
            self.fodNormalize.checked = layer.normalize
            self.fodColorMode.setCurrentIndex(0 if layer.color_mode == "direction" else 1)
            self.fodColor.color = _qcolor(layer.solid_color)
            self.fodOpacity.value = layer.opacity
            self.fodBundleMin.setValue(layer.bundle_min)
            self.fodBundleLobes.checked = layer.bundle_lobes
            self.fodBundleAngle.value = layer.bundle_angle
        finally:
            self._syncing = False
        self.refresh_fod_bundle_combo()
        self.refresh_fod_count()

    def refresh_fod_count(self):
        layer = self.current_fod()
        if layer is None:
            self.fodCount.text = ""
            return
        total = sum(layer.glyph_counts.values())
        self.fodCount.text = "%s glyphes affichés" % layers._fmt_int(total)
        if layer.bundle is not None:
            self.fodBundleInfo.text = layer.bundle_text()

    def on_fod_selected(self, index):
        if not self._syncing:
            self.sync_fod_controls()

    def _set_fod(self, attr, value):
        if self._syncing:
            return
        layer = self.current_fod()
        if layer is None:
            return
        setattr(layer, attr, value)
        layer.apply_display()
        if attr not in ("opacity",):
            layer.invalidate()
            self.logic.request_views()

    def on_remove_fod(self):
        layer = self.current_fod()
        if layer is not None:
            self.logic.remove_fod(layer)

    def on_align_to_fod(self):
        layer = self.current_fod()
        if layer is not None and layer.l0_node is not None:
            for sn in layers.slice_nodes():
                sn.RotateToVolumePlane(layer.l0_node)
            self.logic.request_views()

    def on_l0_background(self):
        layer = self.current_fod()
        if layer is not None and layer.l0_node is not None:
            slicer.util.setSliceViewerLayers(background=layer.l0_node)

    # ---------------------------------------------------- FOD of a bundle
    def refresh_fod_bundle_combo(self):
        self._syncing = True
        try:
            self.fodBundleCombo.clear()
            self.fodBundleCombo.addItem("Aucun (toutes les FOD)")
            for layer in self.logic.tract_layers:
                self.fodBundleCombo.addItem(layer.name)
            fod = self.current_fod()
            index = 0
            if fod is not None and fod.bundle in self.logic.tract_layers:
                index = self.logic.tract_layers.index(fod.bundle) + 1
            self.fodBundleCombo.setCurrentIndex(index)
        finally:
            self._syncing = False
        self._sync_fod_bundle_rows()

    def _sync_fod_bundle_rows(self):
        fod = self.current_fod()
        active = fod is not None and fod.bundle is not None
        self.fodBundleMin.enabled = active
        self.fodBundleLobes.enabled = active
        lobes = active and fod.bundle_lobes
        self.fodBundleAngle.enabled = lobes
        self.fodBundleAngleLabel.enabled = lobes
        self.fodBundleInfo.text = fod.bundle_text() if active else ""
        self.fodBundleInfo.visible = active

    def on_fod_bundle(self, index):
        if self._syncing:
            return
        fod = self.current_fod()
        if fod is None:
            return
        layer = self.logic.tract_layers[index - 1] if 0 < index <= len(self.logic.tract_layers) else None
        with _WaitCursor():
            self.logic.set_fod_bundle(fod, layer)
        self._sync_fod_bundle_rows()

    def on_fod_bundle_lobes(self, on):
        self._set_fod("bundle_lobes", on)
        self._sync_fod_bundle_rows()

    def on_show_bundle_fod(self):
        layer = self.current_tract()
        if layer is None:
            return
        fod = self.current_fod()
        if fod is None:
            slicer.util.warningDisplay("Charge d'abord une image FOD (bouton « FOD »).")
            return
        with _WaitCursor():
            fod.visible = True
            fod.apply_display()
            self.logic.set_fod_bundle(fod, layer)
        self.sync_fod_controls()
        self.fodBox.collapsed = False

    # ------------------------------------------------------------ ROI actions
    def current_rois(self):
        layer = self.current_tract()
        return layer.rois if layer is not None else []

    def on_add_roi(self, shape, mode):
        if self.current_tract() is None:
            slicer.util.warningDisplay("Charge d'abord un tractogramme : les ROI appartiennent à un faisceau.")
            return
        self.logic.add_roi(shape, mode, layer=self.current_tract())

    def on_add_mask(self):
        node = self.maskSelector.currentNode()
        if node is None:
            slicer.util.warningDisplay("Choisis d'abord une segmentation, une labelmap ou un volume.")
            return
        if self.current_tract() is None:
            slicer.util.warningDisplay("Charge d'abord un tractogramme : les ROI appartiennent à un faisceau.")
            return
        with slicer.util.tryWithErrorDisplay("Impossible d'utiliser ce masque", waitCursor=True):
            self.logic.add_mask_roi(node, layer=self.current_tract())

    def on_include_mode(self, all_on):
        if self._syncing:
            return
        layer = self.current_tract()
        if layer is not None:
            self.logic.set_include_mode(layer, "all" if all_on else "any")

    def refresh_roi_table(self):
        self._syncing = True
        try:
            entries = self.current_rois()
            self.roiTable.setRowCount(len(entries))
            for row, e in enumerate(entries):
                swatch = qt.QTableWidgetItem("")
                swatch.setFlags(qt.Qt.ItemIsEnabled)
                swatch.setBackground(qt.QBrush(_qcolor(e.color)))
                self.roiTable.setItem(row, 0, swatch)

                name = qt.QTableWidgetItem(e.name)
                name.setToolTip("%s · double-clic pour centrer les coupes" % e.shape_label)
                if e.shape == "mask":
                    name.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                else:
                    name.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEditable)
                self.roiTable.setItem(row, 1, name)

                mode = qt.QComboBox()
                for key, label in (("include", "Inclure"), ("exclude", "Exclure"), ("endpoints", "Extrémités")):
                    mode.addItem(label, key)
                mode.setCurrentIndex(["include", "exclude", "endpoints"].index(e.mode))
                mode.currentIndexChanged.connect(lambda i, entry=e, combo=mode: self.logic.set_roi_mode(entry, combo.itemData(i)))
                self.roiTable.setCellWidget(row, 2, mode)

                enabled = qt.QCheckBox()
                enabled.checked = e.enabled
                enabled.toggled.connect(lambda on, entry=e: self.logic.set_roi_enabled(entry, on))
                self.roiTable.setCellWidget(row, 3, _centered(enabled))

                actions = qt.QWidget()
                al = qt.QHBoxLayout(actions)
                al.setContentsMargins(2, 0, 2, 0)
                al.setSpacing(2)
                jump = qt.QToolButton()
                jump.setText("◎")
                jump.setToolTip("Centrer les coupes sur cette ROI")
                jump.setAutoRaise(True)
                jump.clicked.connect(lambda checked=False, entry=e: self.logic.jump_to_roi(entry))
                delete = qt.QToolButton()
                trash = _icon("SP_TrashIcon")
                if trash is not None:
                    delete.setIcon(trash)
                else:
                    delete.setText("✕")
                delete.setToolTip("Supprimer cette ROI")
                delete.setAutoRaise(True)
                delete.clicked.connect(lambda checked=False, entry=e: self.logic.remove_roi(entry))
                al.addWidget(jump)
                al.addWidget(delete)
                self.roiTable.setCellWidget(row, 4, actions)
        finally:
            self._syncing = False

    def on_roi_item_changed(self, item):
        if self._syncing or item.column() != 1:
            return
        rois = self.current_rois()
        row = item.row()
        if 0 <= row < len(rois):
            e = rois[row]
            if e.shape != "mask":
                e.node.SetName(item.text())

    def on_live_toggled(self, on):
        self.logic.live = on
        self.applyButton.enabled = not on
        if on:
            self.logic.request_filter()

    def refresh_filter_stats(self):
        self.refresh_bundle_counts()
        layer = self.current_tract()
        if layer is None:
            self.ratioBar.setValue(0)
            self.ratioBar.setFormat("Aucun tractogramme")
            self.filterStats.text = ""
            return
        n_rois = len([e for e in layer.rois if e.enabled])
        total = layer.n_displayed_total
        sel = layer.n_selected
        frac = sel / float(total) if total else 0.0
        self.ratioBar.setValue(int(round(frac * 1000)))
        self.ratioBar.setFormat("%s / %s fibres  (%.1f %%)" % (layers._fmt_int(sel), layers._fmt_int(total), 100 * frac))
        parts = []
        if n_rois == 0:
            parts.append("Aucune ROI active pour ce faisceau : toutes ses fibres sont affichées.")
        if layer.is_subsampled:
            parts.append("Compté sur le sous-échantillon affiché (%s / %s fibres)."
                         % (layers._fmt_int(layer.ws.n_streamlines), layers._fmt_int(layer.full.n_streamlines)))
        if self.logic.last_filter_ms is not None and n_rois:
            parts.append("Filtrage : %.0f ms" % self.logic.last_filter_ms)
        self.filterStats.text = "<br>".join(parts)

    def on_export_selection(self):
        layer = self.current_tract()
        if layer is None:
            return
        base = os.path.splitext(layer.name)[0]
        start = os.path.join(os.path.dirname(layer.path) if layer.path else self._last_dir(), base + "_filtered.tck")
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Enregistrer la sélection", start,
                                              "Tractogrammes (*.tck)")
        if not path:
            return
        if not path.lower().endswith(".tck"):
            path += ".tck"
        with slicer.util.tryWithErrorDisplay("Échec de l'enregistrement", waitCursor=True):
            n = self.logic.export_selection(layer, path)
            slicer.util.showStatusMessage("%s fibres enregistrées dans %s" % (layers._fmt_int(n), os.path.basename(path)), 6000)

    def on_selection_to_layer(self):
        layer = self.current_tract()
        if layer is None:
            return
        with slicer.util.tryWithErrorDisplay("Impossible de créer le tractogramme", waitCursor=True):
            self.logic.selection_to_layer(layer)

    def on_clear_rois(self):
        layer = self.current_tract()
        if layer is not None and layer.rois and slicer.util.confirmYesNoDisplay(
                "Supprimer toutes les ROI de « %s » ?" % layer.name):
            self.logic.clear_rois(layer)

    # ------------------------------------------------------------ logic events
    def _on_logic_event(self, event, payload=None):
        if event == "tracts":
            self.refresh_tract_list(select=payload)
            self.refresh_filter_stats()
            self.sync_fod_controls()
        elif event == "fods":
            self.refresh_fod_list(select=payload)
        elif event == "rgbs":
            self.refresh_rgb_list(select=payload)
        elif event == "rois":
            self.refresh_roi_table()
            self.refresh_filter_stats()
        elif event == "filtered":
            self.refresh_filter_stats()
        elif event == "fod-bundle":
            self._sync_fod_bundle_rows()
        elif event == "views":
            self.refresh_fod_count()


def _count_text(layer):
    total = layer.n_displayed_total
    sel = layer.n_selected
    if sel == total:
        return layers._fmt_int(total)
    return "%s / %s" % (layers._fmt_int(sel), layers._fmt_int(total))


def _swatch_style(layer):
    if layer.color_mode == "solid":
        r, g, b = [int(round(255 * c)) for c in layer.solid_color]
        bg = "rgb(%d,%d,%d)" % (r, g, b)
    else:
        # colour depends on the fibre (direction, scalar...): show a direction-like gradient
        bg = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e74c3c, stop:0.5 #2ecc71, stop:1 #3498db)"
    return "QToolButton { background: %s; border: 1px solid rgba(0,0,0,0.35); border-radius: 4px; }" % bg


class _WaitCursor(object):
    def __enter__(self):
        qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        return self

    def __exit__(self, *args):
        qt.QApplication.restoreOverrideCursor()
        return False


def _centered(widget):
    w = qt.QWidget()
    l = qt.QHBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 0)
    l.addStretch(1)
    l.addWidget(widget)
    l.addStretch(1)
    return w


#
# Logic
#

class MRViewLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)
        self.tract_layers = []
        self.fod_layers = []
        self.rgb_layers = []
        self.active_bundle = None  # bundle whose ROIs are being edited (and shown)
        self._bundle_counter = 1
        self._roi_counter = {}
        self._changed_bundles = set()
        self.live = True
        self.rois_visible = True
        self.last_filter_ms = None
        self._listeners = []
        self._dirty_views = set()
        self._all_views_dirty = False

        self._view_timer = qt.QTimer()
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(10)
        self._view_timer.connect("timeout()", self._process_views)
        self._filter_timer = qt.QTimer()
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(5)
        self._filter_timer.connect("timeout()", self.apply_filters)
        # FODs filtered by a bundle follow its ROIs, throttled so dragging stays fluid
        self._fod_bundle_timer = qt.QTimer()
        self._fod_bundle_timer.setSingleShot(True)
        self._fod_bundle_timer.setInterval(120)
        self._fod_bundle_timer.connect("timeout()", self._update_fod_bundles)

        self._observe_slice_nodes()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self._on_node_added)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAboutToBeRemovedEvent, self._on_node_removed)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self._on_scene_closing)

    def cleanup(self):
        self._view_timer.stop()
        self._filter_timer.stop()
        self._fod_bundle_timer.stop()
        self.removeObservers()

    # -- listeners ---------------------------------------------------------------
    def add_listener(self, fn):
        self._listeners.append(fn)

    def _notify(self, event, payload=None):
        for fn in list(self._listeners):
            try:
                fn(event, payload)
            except Exception as exc:  # never let a GUI refresh break the pipeline
                import traceback
                traceback.print_exc()

    # -- scene observation ---------------------------------------------------------
    def _observe_slice_nodes(self):
        for sn in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            if not self.hasObserver(sn, vtk.vtkCommand.ModifiedEvent, self._on_slice_modified):
                self.addObserver(sn, vtk.vtkCommand.ModifiedEvent, self._on_slice_modified)

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def _on_node_added(self, caller, event, node):
        if node is not None and node.IsA("vtkMRMLSliceNode"):
            self._observe_slice_nodes()
            self.request_views()

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def _on_node_removed(self, caller, event, node):
        if node is None:
            return
        for e in self.all_rois():
            if e.node is node:
                self._drop_roi(e, remove_node=False)
                self._notify("rois")
                self.request_filter()
        for layer in list(self.tract_layers):
            if layer.model3d is node:
                layer.model3d = None
                self.remove_tract(layer)
        for layer in list(self.rgb_layers):
            if layer.node is node:
                layer.node = None
                self.rgb_layers.remove(layer)
                self._notify("rgbs")

    def _on_scene_closing(self, caller=None, event=None):
        for e in self.all_rois():
            self._unobserve_roi(e)
        self.tract_layers = []
        self.fod_layers = []
        self.rgb_layers = []
        self.active_bundle = None
        self._changed_bundles = set()
        self._notify("tracts")
        self._notify("fods")
        self._notify("rgbs")
        self._notify("rois")
        qt.QTimer.singleShot(0, self._observe_slice_nodes)

    def _on_slice_modified(self, caller, event):
        self._dirty_views.add(caller.GetID())
        if not self._view_timer.isActive():
            self._view_timer.start()

    def request_views(self):
        self._all_views_dirty = True
        if not self._view_timer.isActive():
            self._view_timer.start()

    def _process_views(self):
        nodes = layers.slice_nodes()
        if not self._all_views_dirty:
            nodes = [n for n in nodes if n.GetID() in self._dirty_views]
        self._dirty_views = set()
        self._all_views_dirty = False
        if not nodes:
            return
        for sn in nodes:
            for layer in self.tract_layers:
                try:
                    layer.update_slab(sn)
                except Exception:
                    import traceback
                    traceback.print_exc()
            for fod in self.fod_layers:
                try:
                    fod.update_view(sn)
                except Exception:
                    import traceback
                    traceback.print_exc()
        if self.fod_layers:
            self._notify("views")

    # -- loading ---------------------------------------------------------------------
    def load_tractogram(self, path, name=None):
        data = mrtrix_io.read_tck(path)
        name = name or os.path.basename(path)
        t = tracts.Tractogram(data["points"], data["offsets"], data["header"], name)
        return self.add_tractogram(t, name, path)

    def add_tractogram(self, tractogram, name, path=None, max_display=150000):
        first = not self.tract_layers and not self.fod_layers
        layer = layers.TractLayer(tractogram, name, path, max_display=max_display)
        layer.solid_color = self.next_bundle_color()
        cmap = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeFileViridis.txt")
        layer.colormap_id = cmap.GetID() if cmap is not None else None
        self.tract_layers.append(layer)
        qt.QTimer.singleShot(0, layer.apply_tubes)
        self.active_bundle = layer
        self._update_roi_visibility()
        self.request_views()
        if first:
            self._center_on_tracts(layer)
        self._notify("tracts", layer)
        return layer

    def _center_on_tracts(self, layer):
        if layer.ws.n_points == 0:
            return
        has_bg = False
        lm = slicer.app.layoutManager()
        for name in lm.sliceViewNames():
            if lm.sliceWidget(name).sliceLogic().GetSliceCompositeNode().GetBackgroundVolumeID():
                has_bg = True
        center = np.median(layer.ws.points[:: max(1, layer.ws.n_points // 50000)], axis=0)
        if not has_bg:
            lo, hi = layer.ws.points.min(0), layer.ws.points.max(0)
            fov = float(np.max(hi - lo)) * 1.3 + 10
            for sn in layers.slice_nodes():
                sn.JumpSliceByCentering(*[float(x) for x in center])
                w, h = sn.GetDimensions()[:2]
                w, h = max(w, 1), max(h, 1)
                fx, fy = (fov * w / h, fov) if w >= h else (fov, fov * h / w)
                sn.SetFieldOfView(fx, fy, sn.GetFieldOfView()[2])
        else:
            for sn in layers.slice_nodes():
                sn.JumpSliceByOffsetting(*[float(x) for x in center])
        self.reset_3d_camera()

    def reset_3d_camera(self):
        lm = slicer.app.layoutManager()
        for i in range(lm.threeDViewCount):
            view = lm.threeDWidget(i).threeDView()
            view.resetFocalPoint()
            view.resetCamera()

    def recenter_views(self):
        slicer.util.resetSliceViews()
        self.reset_3d_camera()

    def load_tsf(self, layer, path):
        data = mrtrix_io.read_tsf(path)
        if data["offsets"].size - 1 != layer.full.n_streamlines:
            raise ValueError("Le fichier .tsf contient %d fibres, le tractogramme %d."
                             % (data["offsets"].size - 1, layer.full.n_streamlines))
        layer.set_tsf(data["values"], os.path.basename(path))

    def load_fod(self, path):
        image = mrtrix_io.read_image(path)
        if not mrtrix_io.is_sh_image(image):
            raise ValueError("%s n'est pas une image d'harmoniques sphériques (4e dimension = %s)."
                             % (os.path.basename(path), image.data.shape[3:] or "absente"))
        layer = layers.FODLayer(image, os.path.basename(path), path)
        del image
        self.fod_layers.append(layer)
        self.request_views()
        self._notify("fods", layer)
        return layer

    def load_rgb(self, path):
        image = mrtrix_io.read_image(path)
        if not mrtrix_io.is_vector_image(image):
            raise ValueError("%s n'est pas une carte de vecteurs : il faut 3 volumes (dimensions %s)."
                             % (os.path.basename(path), "×".join(str(d) for d in image.data.shape)))
        previous = self._background_volume()
        layer = layers.RGBLayer(image, os.path.basename(path), path)
        del image
        layer.previous_background = previous
        layer.mode = "overlay" if previous is not None else "background"
        layer.opacity = 0.6
        layer.visible = True
        self.rgb_layers.append(layer)
        self.show_rgb(layer, fit=True)
        self.request_views()
        self._notify("rgbs", layer)
        return layer

    def show_rgb(self, layer, fit=False):
        """Background: the RGB map alone. Overlay: on top of the previous background (T1, ...)."""
        if layer.node is None:
            return
        if not layer.visible:
            if self._foreground_volume() is layer.node:
                slicer.util.setSliceViewerLayers(foreground=None)
            if self._background_volume() is layer.node:
                bg = layer.previous_background
                slicer.util.setSliceViewerLayers(background=bg if bg is not None and bg.GetScene() is not None else None)
            return
        if layer.mode == "overlay":
            bg = layer.previous_background
            if bg is not None and (bg.GetScene() is None or bg is layer.node):
                bg = None
            kwargs = {"foreground": layer.node, "foregroundOpacity": layer.opacity}
            if bg is not None:
                kwargs["background"] = bg
            slicer.util.setSliceViewerLayers(fit=fit, **kwargs)
        else:
            if self._foreground_volume() is layer.node:
                slicer.util.setSliceViewerLayers(foreground=None)
            slicer.util.setSliceViewerLayers(background=layer.node, fit=fit)

    def remove_rgb(self, layer):
        if layer in self.rgb_layers:
            self.rgb_layers.remove(layer)
        layer.remove()
        self._notify("rgbs")

    def _composite(self):
        lm = slicer.app.layoutManager()
        names = list(lm.sliceViewNames()) if lm is not None else []
        if not names:
            return None
        return lm.sliceWidget(names[0]).sliceLogic().GetSliceCompositeNode()

    def _background_volume(self):
        comp = self._composite()
        vid = comp.GetBackgroundVolumeID() if comp is not None else None
        return slicer.mrmlScene.GetNodeByID(vid) if vid else None

    def _foreground_volume(self):
        comp = self._composite()
        vid = comp.GetForegroundVolumeID() if comp is not None else None
        return slicer.mrmlScene.GetNodeByID(vid) if vid else None

    def load_image(self, path):
        if mrtrix_io.looks_like_vector(path):
            return self.load_rgb(path)
        low = path.lower()
        if low.endswith((".mif", ".mif.gz", ".mih")):
            image = mrtrix_io.read_mif(path)
            node = layers.add_volume_from_image(image, os.path.basename(path).split(".")[0], 0, show=True)
        else:
            node = slicer.util.loadVolume(path)
            slicer.util.setSliceViewerLayers(background=node, fit=True)
        self.request_views()
        return node

    def set_max_display(self, layer, value):
        layer.set_max_display(value)
        self.apply_filters()
        layer.invalidate_slabs()
        self.request_views()
        self._notify("filtered")

    def remove_tract(self, layer):
        if layer in self.tract_layers:
            self.tract_layers.remove(layer)
        for e in list(layer.rois):
            self._drop_roi(e, remove_node=True)
        if self.active_bundle is layer:
            self.active_bundle = None
        for fod in self.fod_layers:
            if fod.bundle is layer:
                fod.bundle = None
                fod.bundle_map = None
                fod.invalidate()
        layer.remove()
        self.request_views()
        self._notify("tracts")

    def remove_fod(self, layer):
        if layer in self.fod_layers:
            self.fod_layers.remove(layer)
        layer.remove()
        self._notify("fods")

    # -- bundles ---------------------------------------------------------------------
    def next_bundle_color(self):
        used = set(tuple(l.solid_color) for l in self.tract_layers)
        for c in layers.BUNDLE_COLORS:
            if tuple(c) not in used:
                return c
        return layers.BUNDLE_COLORS[len(self.tract_layers) % len(layers.BUNDLE_COLORS)]

    def new_bundle(self, source, copy_rois=False, name=None):
        """A new bundle on the same streamlines as `source`, with its own ROIs and colour.

        copy_rois: also copy the ROIs of `source` (new, independent ROI nodes)."""
        self._bundle_counter += 1
        if name is None:
            name = ("%s (copie)" % source.name) if copy_rois else "Faisceau %d" % self._bundle_counter
        layer = layers.TractLayer(source.full, name, source.path, max_display=source.max_display)
        layer.colormap_id = source.colormap_id
        layer.full_tsf, layer.tsf_name = source.full_tsf, source.tsf_name
        layer.solid_color = self.next_bundle_color()
        layer.color_mode = "solid"
        layer.line_width = source.line_width
        layer.tubes = source.tubes
        layer.slab_mm = source.slab_mm
        layer.include_mode = source.include_mode if copy_rois else "all"
        layer.apply_display()
        self.tract_layers.append(layer)
        qt.QTimer.singleShot(0, layer.apply_tubes)
        if copy_rois:
            for e in source.rois:
                self._copy_roi(e, layer)
        self.active_bundle = layer
        self._update_roi_visibility()
        self.apply_filters()
        self.request_views()
        self._notify("tracts", layer)
        return layer

    def set_active_bundle(self, layer):
        """The bundle being edited: only its ROIs are shown in the views."""
        if layer is self.active_bundle:
            return
        self.active_bundle = layer
        self._update_roi_visibility()

    def set_fod_bundle(self, fod, layer):
        fod.bundle = layer
        fod.compute_bundle_map()
        fod.invalidate()
        self.request_views()
        self._notify("fods", fod)

    def _request_fod_bundles(self, changed):
        if any(f.bundle in changed for f in self.fod_layers):
            self._changed_bundles.update(changed)
            if not self._fod_bundle_timer.isActive():
                self._fod_bundle_timer.start()

    def _update_fod_bundles(self):
        changed, self._changed_bundles = self._changed_bundles, set()
        updated = False
        for fod in self.fod_layers:
            if fod.bundle is not None and fod.bundle in changed:
                try:
                    fod.compute_bundle_map()
                    updated = True
                except Exception:
                    import traceback
                    traceback.print_exc()
        if updated:
            self.request_views()
            self._notify("fod-bundle")

    # -- ROIs ------------------------------------------------------------------------
    def all_rois(self):
        return [e for layer in self.tract_layers for e in layer.rois]

    def _owner(self, entry):
        for layer in self.tract_layers:
            if entry in layer.rois:
                return layer
        return None

    def _roi_should_show(self, entry):
        return self.rois_visible and entry.enabled and self._owner(entry) is self.active_bundle

    def _update_roi_visibility(self):
        for e in self.all_rois():
            e.set_visible(self._roi_should_show(e))

    def _default_roi_center(self, layer=None):
        center = layers.slice_intersection_point()
        layer = layer or (self.tract_layers[0] if self.tract_layers else None)
        if layer is not None:
            pts = layer.ws.points
            if pts.shape[0]:
                lo, hi = pts.min(0), pts.max(0)
                if center is None or np.any(center < lo - 5) or np.any(center > hi + 5):
                    center = np.median(pts[:: max(1, pts.shape[0] // 50000)], axis=0)
        return center if center is not None else np.zeros(3)

    def add_roi(self, shape, mode="include", center=None, size=None, layer=None):
        layer = layer or self.active_bundle or (self.tract_layers[-1] if self.tract_layers else None)
        if layer is None:
            raise ValueError("Charge d'abord un tractogramme : les ROI appartiennent à un faisceau.")
        self._roi_counter[shape] = self._roi_counter.get(shape, 0) + 1
        prefix = {"box": "Boîte", "sphere": "Sphère"}[shape]
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "%s %d" % (prefix, self._roi_counter[shape]))
        node.CreateDefaultDisplayNodes()
        center = self._default_roi_center(layer) if center is None else np.asarray(center, float)
        size = size if size is not None else (10.0 if shape == "box" else 8.0)
        size = [float(size)] * 3 if np.isscalar(size) else [float(x) for x in size]
        node.SetCenter([float(x) for x in center])
        node.SetSize(size)
        node.SetAttribute("MRView.Shape", shape)
        layers.put_in_folder(node)
        entry = layers.ROIEntry(node, shape, mode)
        layer.rois.append(entry)
        entry.apply_color()
        entry.update_sphere_model()
        entry.set_visible(self._roi_should_show(entry))
        self._observe_roi(entry)
        self._notify("rois")
        self.apply_filters()
        return entry

    def _copy_roi(self, entry, layer):
        if entry.shape == "mask":
            new = layers.ROIEntry(entry.node, "mask", entry.mode, segment_id=entry.segment_id)
            new.load_mask()
            new.enabled = entry.enabled
            layer.rois.append(new)
            new.apply_color()
            self._observe_roi(new)
            return new
        c = [0.0, 0.0, 0.0]
        entry.node.GetCenter(c)
        new = self.add_roi(entry.shape, entry.mode, center=c, size=list(entry.node.GetSize()), layer=layer)
        try:
            m = vtk.vtkMatrix4x4()
            m.DeepCopy(entry.node.GetObjectToNodeMatrix())
            new.node.SetAndObserveObjectToNodeMatrix(m)
            new.node.SetCenter(c)
        except Exception:
            pass
        if entry.node.GetParentTransformNode() is not None:
            new.node.SetAndObserveTransformNodeID(entry.node.GetParentTransformNode().GetID())
        new.update_sphere_model()
        new.enabled = entry.enabled
        new.set_visible(self._roi_should_show(new))
        return new

    def add_mask_roi(self, node, mode="include", layer=None):
        layer = layer or self.active_bundle or (self.tract_layers[-1] if self.tract_layers else None)
        if layer is None:
            raise ValueError("Charge d'abord un tractogramme : les ROI appartiennent à un faisceau.")
        added = []
        if node.IsA("vtkMRMLSegmentationNode"):
            seg = node.GetSegmentation()
            for i in range(seg.GetNumberOfSegments()):
                sid = seg.GetNthSegmentID(i)
                entry = layers.ROIEntry(node, "mask", mode, segment_id=sid)
                entry.load_mask()
                added.append(entry)
        else:
            entry = layers.ROIEntry(node, "mask", mode)
            entry.load_mask()
            added.append(entry)
        if not added:
            raise ValueError("Cette segmentation ne contient aucun segment.")
        for entry in added:
            entry.apply_color()
            self._observe_roi(entry)
            layer.rois.append(entry)
        self._notify("rois")
        self.apply_filters()
        return added

    def _observe_roi(self, entry):
        node = entry.node
        if entry.shape in ("box", "sphere"):
            for ev in (slicer.vtkMRMLMarkupsNode.PointModifiedEvent, vtk.vtkCommand.ModifiedEvent):
                entry.observer_tags.append((node, node.AddObserver(ev, lambda c, e, entry=entry: self._on_roi_moved(entry))))
            ev_end = getattr(slicer.vtkMRMLMarkupsNode, "PointEndInteractionEvent", None)
            if ev_end is not None:
                entry.observer_tags.append((node, node.AddObserver(ev_end, lambda c, e: self._on_roi_released())))
        elif node.IsA("vtkMRMLSegmentationNode"):
            seg = node.GetSegmentation()
            for name in ("SourceRepresentationModified", "MasterRepresentationModified"):
                ev = getattr(slicer.vtkSegmentation, name, None)
                if ev is not None:
                    entry.observer_tags.append((seg, seg.AddObserver(ev, lambda c, e, entry=entry: self._on_mask_modified(entry))))
                    break
        else:
            ev = slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent
            entry.observer_tags.append((node, node.AddObserver(ev, lambda c, e, entry=entry: self._on_mask_modified(entry))))

    def _unobserve_roi(self, entry):
        for obj, tag in entry.observer_tags:
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
        entry.observer_tags = []

    def _on_roi_moved(self, entry):
        if entry.shape == "sphere":
            entry.update_sphere_model()
        if self.live:
            self.request_filter()

    def _on_roi_released(self):
        if not self.live:
            self.request_filter()

    def _on_mask_modified(self, entry):
        entry._mask = None
        entry.version += 1
        if self.live:
            self.request_filter()

    def set_roi_mode(self, entry, mode):
        entry.mode = mode
        entry.apply_color()
        self._notify("rois")
        self.apply_filters()

    def set_roi_enabled(self, entry, enabled):
        entry.enabled = enabled
        entry.set_visible(self._roi_should_show(entry))
        self.apply_filters()

    def set_rois_visible(self, visible):
        self.rois_visible = visible
        self._update_roi_visibility()

    def set_include_mode(self, layer, mode):
        layer.include_mode = mode
        self.apply_filters()

    def _drop_roi(self, entry, remove_node=True):
        self._unobserve_roi(entry)
        for layer in self.tract_layers:
            if entry in layer.rois:
                layer.rois.remove(entry)
            for key in [k for k in layer.roi_cache if k[0] == entry.uid]:
                del layer.roi_cache[key]
        entry.remove(remove_node=remove_node)

    def remove_roi(self, entry):
        if isinstance(entry, int):
            rois = self.active_bundle.rois if self.active_bundle is not None else []
            if not (0 <= entry < len(rois)):
                return
            entry = rois[entry]
        self._drop_roi(entry, remove_node=True)
        self._notify("rois")
        self.apply_filters()

    def clear_rois(self, layer=None):
        targets = [layer] if layer is not None else list(self.tract_layers)
        for l in targets:
            for e in list(l.rois):
                self._drop_roi(e, remove_node=True)
        self._notify("rois")
        self.apply_filters()

    def jump_to_roi(self, entry):
        if isinstance(entry, int):
            rois = self.active_bundle.rois if self.active_bundle is not None else []
            if not (0 <= entry < len(rois)):
                return
            entry = rois[entry]
        if entry.shape == "mask":
            mask, A = entry._mask, entry._ras_to_ijk
            if mask is None:
                entry.load_mask()
                mask, A = entry._mask, entry._ras_to_ijk
            ijk = np.argwhere(mask)
            if not ijk.size:
                return
            c_ijk = ijk.mean(0)
            Ainv = np.linalg.inv(A)
            center = Ainv[:3, :3] @ c_ijk + Ainv[:3, 3]
        else:
            c = [0.0, 0.0, 0.0]
            entry.node.GetCenterWorld(c)
            center = c
        for sn in layers.slice_nodes():
            sn.JumpSliceByCentering(*[float(x) for x in center])

    # -- filtering -------------------------------------------------------------------
    def request_filter(self):
        if not self._filter_timer.isActive():
            self._filter_timer.start()

    def apply_filters(self):
        """Each bundle is filtered by its own ROIs."""
        t0 = time.time()
        changed = set()
        for layer in self.tract_layers:
            incl, excl = [], []
            for e in [e for e in layer.rois if e.enabled]:
                key = (e.uid, e.mode == "endpoints")
                try:
                    sig = e.signature()
                except Exception:
                    continue
                cached = layer.roi_cache.get(key)
                if cached is None or cached[0] != sig:
                    hits = tracts.streamline_hits(layer.ws, e.point_test(), endpoints_only=(e.mode == "endpoints"))
                    layer.roi_cache[key] = (sig, hits)
                else:
                    hits = cached[1]
                (excl if e.mode == "exclude" else incl).append(hits)
            sel = tracts.combine_selection(layer.ws.n_streamlines, incl, excl, layer.include_mode)
            if layer.set_selection(sel):
                layer.invalidate_slabs()
                changed.add(layer)
        self.last_filter_ms = 1000.0 * (time.time() - t0)
        # update 2D slabs right away so 2D and 3D stay in sync while dragging
        self._all_views_dirty = True
        self._process_views()
        self._request_fod_bundles(changed)
        self._notify("filtered")

    def full_selection(self, layer):
        return layer.selection_for_full()

    def export_selection(self, layer, path):
        sel = self.full_selection(layer)
        sub = layer.full.subset(np.flatnonzero(sel))
        header = {k: v for k, v in layer.full.header.items() if k not in ("count", "total_count", "file", "datatype", "timestamp")}
        mrtrix_io.write_tck(path, sub.points, sub.offsets, header)
        return sub.n_streamlines

    def selection_to_layer(self, layer):
        sel = self.full_selection(layer)
        sub = layer.full.subset(np.flatnonzero(sel))
        name = os.path.splitext(layer.name)[0] + " (sélection)"
        new = self.add_tractogram(sub, name, None, max_display=layer.max_display)
        new.color_mode = layer.color_mode
        new.solid_color = layer.solid_color
        new.apply_display()
        self._notify("tracts", new)
        return new


#
# Test
#

class MRViewTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_synthetic_pipeline()

    def test_synthetic_pipeline(self):
        import tempfile
        self.delayDisplay("Création de données synthétiques")
        rng = np.random.RandomState(0)
        lengths = rng.randint(20, 60, size=300)
        starts = rng.uniform(-30, 30, size=(300, 3))
        dirs = rng.randn(300, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = np.concatenate([s + np.outer(np.arange(n), d) for s, d, n in zip(starts, dirs, lengths)]).astype(np.float32)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        tmp = tempfile.mkdtemp()
        tck = os.path.join(tmp, "synthetic.tck")
        mrtrix_io.write_tck(tck, pts, offsets)

        # small lmax 4 FOD with a single fibre along x
        from MRViewLib import sh
        coefs = sh.sh_basis(np.array([[1.0, 0, 0]]), 4)[0]
        data = np.zeros((10, 10, 10, 15), np.float32)
        data[2:8, 2:8, 2:8] = coefs
        M = np.diag([2.0, 2.0, 2.0, 1.0])
        M[:3, 3] = -10
        fod = os.path.join(tmp, "fod.mif")
        mrtrix_io.write_mif(fod, data, M)

        logic = MRViewLogic()
        layer = logic.load_tractogram(tck)
        self.assertEqual(layer.full.n_streamlines, 300)
        fod_layer = logic.load_fod(fod)
        self.assertEqual(fod_layer.lmax, 4)
        logic._process_views()
        self.assertTrue(sum(fod_layer.glyph_counts.values()) > 0)

        for mode, _ in COLOR_MODES[:5]:
            layer.color_mode = mode
            layer.apply_display()

        self.delayDisplay("ROI")
        entry = logic.add_roi("sphere", "include", center=pts[0], size=6.0)
        self.assertTrue(0 < layer.n_selected < 300)
        n_incl = layer.n_selected
        logic.set_roi_mode(entry, "exclude")
        self.assertEqual(layer.n_selected, 300 - n_incl)
        out = os.path.join(tmp, "filtered.tck")
        n = logic.export_selection(layer, out)
        self.assertEqual(n, 300 - n_incl)
        self.assertEqual(mrtrix_io.read_tck(out)["offsets"].size - 1, n)
        logic.cleanup()
        self.delayDisplay("Test réussi")
