"""
Module d'exportation d'animations vers des formats standards (glTF, FBX).
"""

import numpy as np
import trimesh
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
import pygltflib
from pygltflib import GLTF2, Node, Scene, Buffer, BufferView, Accessor
from pygltflib import Animation, AnimationChannel, AnimationSampler
import struct
import base64
import fbx
import FbxCommon
from .text_animator import TextAnimator, AnimationClip, AnimationKeyframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExportSettings:
    """Paramètres d'exportation."""

    format: str = "gltf"  # "gltf" ou "fbx"
    embed_textures: bool = True
    compression: bool = True
    binary: bool = True  # Pour glTF, utiliser .glb au lieu de .gltf+.bin
    optimize_keyframes: bool = True
    fps: int = 60


class AnimationExporter:
    """Exporteur d'animations vers différents formats."""

    def __init__(self, settings: Optional[ExportSettings] = None):
        self.settings = settings or ExportSettings()

    def export_animation(
        self,
        mesh: trimesh.Trimesh,
        animator: TextAnimator,
        output_path: Union[str, Path],
    ):
        """
        Exporte une animation vers le format spécifié.

        Args:
            mesh: Maillage de base
            animator: Animateur contenant les animations
            output_path: Chemin de sortie pour le fichier
        """
        output_path = Path(output_path)

        if self.settings.format.lower() == "gltf":
            self._export_gltf(mesh, animator, output_path)
        elif self.settings.format.lower() == "fbx":
            self._export_fbx(mesh, animator, output_path)
        else:
            raise ValueError(f"Format non supporté: {self.settings.format}")

    def _optimize_keyframes(
        self, keyframes: List[AnimationKeyframe], tolerance: float = 1e-4
    ) -> List[AnimationKeyframe]:
        """Optimise les keyframes en supprimant celles qui sont redondantes."""
        if len(keyframes) <= 2:
            return keyframes

        result = [keyframes[0]]

        for i in range(1, len(keyframes) - 1):
            prev = keyframes[i - 1]
            curr = keyframes[i]
            next_kf = keyframes[i + 1]

            # Calculer l'interpolation linéaire
            t = (curr.time - prev.time) / (next_kf.time - prev.time)
            expected = prev.value + t * (next_kf.value - prev.value)

            # Si la différence est supérieure au seuil, garder la keyframe
            if np.max(np.abs(curr.value - expected)) > tolerance:
                result.append(curr)

        result.append(keyframes[-1])
        return result

    def _export_gltf(
        self, mesh: trimesh.Trimesh, animator: TextAnimator, output_path: Path
    ):
        """Exporte l'animation au format glTF."""

        # Créer un nouveau document glTF
        gltf = GLTF2()

        # Ajouter une scène par défaut
        scene = Scene(nodes=[0])
        gltf.scenes.append(scene)
        gltf.scene = 0

        # Créer un noeud pour le maillage
        node = Node(mesh=0)
        gltf.nodes.append(node)

        # Convertir le maillage
        vertices = mesh.vertices.flatten().tobytes()
        indices = mesh.faces.flatten().astype(np.uint32).tobytes()

        # Créer le buffer pour les données géométriques
        buffer_data = vertices + indices
        if self.settings.binary:
            gltf.buffers.append(Buffer(byteLength=len(buffer_data)))
        else:
            buffer_uri = base64.b64encode(buffer_data).decode("ascii")
            gltf.buffers.append(
                Buffer(
                    uri=f"data:application/octet-stream;base64,{buffer_uri}",
                    byteLength=len(buffer_data),
                )
            )

        # Créer les buffer views
        vertex_view = BufferView(
            buffer=0,
            byteOffset=0,
            byteLength=len(vertices),
            target=pygltflib.ARRAY_BUFFER,
        )
        index_view = BufferView(
            buffer=0,
            byteOffset=len(vertices),
            byteLength=len(indices),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
        gltf.bufferViews.extend([vertex_view, index_view])

        # Créer les accesseurs
        vertex_accessor = Accessor(
            bufferView=0,
            componentType=pygltflib.FLOAT,
            count=len(mesh.vertices),
            type="VEC3",
            max=mesh.vertices.max(axis=0).tolist(),
            min=mesh.vertices.min(axis=0).tolist(),
        )
        index_accessor = Accessor(
            bufferView=1,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(mesh.faces.flatten()),
            type="SCALAR",
        )
        gltf.accessors.extend([vertex_accessor, index_accessor])

        # Ajouter les animations
        animation_buffer_data = bytearray()
        current_offset = len(buffer_data)

        for anim_name, clip in animator.animations.items():
            keyframes = clip.keyframes
            if self.settings.optimize_keyframes:
                keyframes = self._optimize_keyframes(keyframes)

            # Créer les buffer views pour les temps et valeurs
            times = np.array([kf.time for kf in keyframes], dtype=np.float32)
            values = np.array([kf.value for kf in keyframes], dtype=np.float32)

            time_data = times.tobytes()
            value_data = values.tobytes()

            time_view = BufferView(
                buffer=0, byteOffset=current_offset, byteLength=len(time_data)
            )
            value_view = BufferView(
                buffer=0,
                byteOffset=current_offset + len(time_data),
                byteLength=len(value_data),
            )
            gltf.bufferViews.extend([time_view, value_view])

            # Créer les accesseurs
            time_accessor = Accessor(
                bufferView=len(gltf.bufferViews) - 2,
                componentType=pygltflib.FLOAT,
                count=len(times),
                type="SCALAR",
                min=[times.min()],
                max=[times.max()],
            )
            value_accessor = Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=pygltflib.FLOAT,
                count=len(values),
                type="VEC3",
                min=values.min(axis=0).tolist(),
                max=values.max(axis=0).tolist(),
            )
            gltf.accessors.extend([time_accessor, value_accessor])

            # Créer le sampler et le channel
            sampler = AnimationSampler(
                input=len(gltf.accessors) - 2,
                output=len(gltf.accessors) - 1,
                interpolation="LINEAR",
            )

            channel = AnimationChannel(
                sampler=len(gltf.animations[-1].samplers) if gltf.animations else 0,
                target={
                    "node": 0,
                    "path": "translation",  # ou "rotation", "scale" selon le type
                },
            )

            # Créer l'animation
            animation = Animation(
                name=anim_name, channels=[channel], samplers=[sampler]
            )
            gltf.animations.append(animation)

            # Mettre à jour le buffer
            animation_buffer_data.extend(time_data)
            animation_buffer_data.extend(value_data)
            current_offset += len(time_data) + len(value_data)

        # Mettre à jour le buffer principal
        if self.settings.binary:
            with open(output_path.with_suffix(".glb"), "wb") as f:
                gltf.save_binary(f, buffer_data + animation_buffer_data)
        else:
            gltf.buffers[0].uri = (
                f"data:application/octet-stream;base64,{base64.b64encode(buffer_data + animation_buffer_data).decode('ascii')}"
            )
            gltf.save_json(output_path.with_suffix(".gltf"))

    def _export_fbx(
        self, mesh: trimesh.Trimesh, animator: TextAnimator, output_path: Path
    ):
        """Exporte l'animation au format FBX."""

        # Initialiser le SDK FBX
        manager = fbx.FbxManager.Create()
        if not manager:
            raise RuntimeError("Erreur lors de la création du FBxManager")

        ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
        manager.SetIOSettings(ios)

        # Créer une nouvelle scène
        scene = fbx.FbxScene.Create(manager, "Scene")

        # Créer le noeud racine
        root_node = scene.GetRootNode()

        # Créer un noeud pour le maillage
        mesh_node = fbx.FbxNode.Create(scene, "TextMesh")
        root_node.AddChild(mesh_node)

        # Créer le maillage FBX
        fbx_mesh = fbx.FbxMesh.Create(scene, "TextMeshGeometry")

        # Ajouter les vertices
        fbx_mesh.InitControlPoints(len(mesh.vertices))
        for i, vertex in enumerate(mesh.vertices):
            fbx_mesh.SetControlPointAt(fbx.FbxVector4(*vertex, 0.0), i)

        # Ajouter les faces
        for face in mesh.faces:
            fbx_mesh.BeginPolygon()
            for vertex_index in face:
                fbx_mesh.AddPolygon(vertex_index)
            fbx_mesh.EndPolygon()

        mesh_node.SetNodeAttribute(fbx_mesh)

        # Créer l'animation
        animation_stack = fbx.FbxAnimStack.Create(scene, "Animation")
        animation_layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
        animation_stack.AddMember(animation_layer)

        # Créer les courbes d'animation
        for anim_name, clip in animator.animations.items():
            keyframes = clip.keyframes
            if self.settings.optimize_keyframes:
                keyframes = self._optimize_keyframes(keyframes)

            # Créer les courbes de translation
            translation_curve_x = mesh_node.LclTranslation.GetCurve(
                animation_layer, "X", True
            )
            translation_curve_y = mesh_node.LclTranslation.GetCurve(
                animation_layer, "Y", True
            )
            translation_curve_z = mesh_node.LclTranslation.GetCurve(
                animation_layer, "Z", True
            )

            # Ajouter les keyframes
            for kf in keyframes:
                time = fbx.FbxTime()
                time.SetSecondDouble(kf.time)

                translation_curve_x.KeyModifyBegin()
                key_index = translation_curve_x.KeyAdd(time)[0]
                translation_curve_x.KeySetValue(key_index, kf.value[0])
                translation_curve_x.KeySetInterpolation(
                    key_index, fbx.FbxAnimCurveDef.eInterpolationLinear
                )
                translation_curve_x.KeyModifyEnd()

                translation_curve_y.KeyModifyBegin()
                key_index = translation_curve_y.KeyAdd(time)[0]
                translation_curve_y.KeySetValue(key_index, kf.value[1])
                translation_curve_y.KeySetInterpolation(
                    key_index, fbx.FbxAnimCurveDef.eInterpolationLinear
                )
                translation_curve_y.KeyModifyEnd()

                translation_curve_z.KeyModifyBegin()
                key_index = translation_curve_z.KeyAdd(time)[0]
                translation_curve_z.KeySetValue(key_index, kf.value[2])
                translation_curve_z.KeySetInterpolation(
                    key_index, fbx.FbxAnimCurveDef.eInterpolationLinear
                )
                translation_curve_z.KeyModifyEnd()

        # Créer l'exporteur et sauvegarder
        exporter = fbx.FbxExporter.Create(manager, "")
        if not exporter.Initialize(
            str(output_path.with_suffix(".fbx")), -1, manager.GetIOSettings()
        ):
            raise RuntimeError("Impossible d'initialiser l'exporteur FBX")

        exporter.Export(scene)

        # Nettoyer
        exporter.Destroy()
        scene.Destroy()
        manager.Destroy()
