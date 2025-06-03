# -*- coding: utf-8 -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *

import __main__


import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

import json
import os

def run_sim(report_path,
            largeur, #mm
            hauteur, #mm
            distance_x, #mm
            distance_y,
            rayon, #mm
            plaque_epaisseur, #mm
            frequ_max_mode,
            materiau,
            #mesh params
            elem_size,
            deviationFactor,
            minSizeFactor
            ) : #vérifier unité si Hz ou rad/s?


    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
        sheetSize=500.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Spot(point=(0.0, 0.0))
    s.FixedConstraint(entity=v[0])
    s.rectangle(point1=(0.0, 0.0), point2=(largeur, hauteur))


    s.CircleByCenterPerimeter(center=(distance_x, distance_y), point1=(distance_x, distance_y + rayon))
    s.CircleByCenterPerimeter(center=(distance_x, hauteur - distance_y), point1=(distance_x, hauteur - distance_y - rayon))
    s.CircleByCenterPerimeter(center=(largeur - distance_x, hauteur - distance_y), point1=(largeur - distance_x, hauteur - distance_y - rayon))
    s.CircleByCenterPerimeter(center=(largeur - distance_x, distance_y), point1=(largeur - distance_x, distance_y + rayon))


    p = mdb.models['Model-1'].Part(name='plaque', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['plaque']
    p.BaseShell(sketch=s)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['plaque']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
        engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    mdb.models['Model-1'].Material(name=materiau["name"])
    mdb.models['Model-1'].materials[materiau["name"]].Density(table=((materiau["density"],), ))
    mdb.models['Model-1'].materials[materiau["name"]].Elastic(table=((materiau["young_modulus"], materiau["poisson_modulus"]), ))
    mdb.models['Model-1'].HomogeneousShellSection(name='tole', preIntegrate=OFF, 
        material=materiau["name"], thicknessType=UNIFORM, thickness=plaque_epaisseur, 
        thicknessField='', nodalThicknessField='', 
        idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
        thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
        integrationRule=SIMPSON, numIntPts=5)
    p = mdb.models['Model-1'].parts['plaque']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['plaque']
    p.SectionAssignment(region=region, sectionName='tole', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['plaque']
    a.Instance(name='plaque-1', part=p, dependent=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)
    mdb.models['Model-1'].FrequencyStep(name='analyse_modale', previous='Initial', 
        maxEigen=frequ_max_mode)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        step='analyse_modale')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
        constraints=ON, connectors=ON, engineeringFeatures=ON, 
        adaptiveMeshConstraints=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, interactions=OFF, constraints=OFF, connectors=ON, 
        engineeringFeatures=OFF)
    a = mdb.models['Model-1'].rootAssembly
    e1 = a.instances['plaque-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#f ]', ), )
    region = regionToolset.Region(edges=edges1)
    mdb.models['Model-1'].EncastreBC(name='encastre', 
        createStepName='analyse_modale', region=region, localCsys=None)
        
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
        bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['Model-1'].parts['plaque']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
        engineeringFeatures=OFF, mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['Model-1'].parts['plaque']
    p.seedPart(size=elem_size, deviationFactor=deviationFactor, minSizeFactor=minSizeFactor)
    p = mdb.models['Model-1'].parts['plaque']
    p.generateMesh()
    p = mdb.models['Model-1'].parts['plaque']
    f = p.faces
    pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.deleteMesh(regions=pickedRegions)
    p = mdb.models['Model-1'].parts['plaque']
    f = p.faces
    pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.setMeshControls(regions=pickedRegions, algorithm=MEDIAL_AXIS)
    p = mdb.models['Model-1'].parts['plaque']
    p.generateMesh()
    p = mdb.models['Model-1'].parts['plaque']
    f = p.faces
    pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.deleteMesh(regions=pickedRegions)
    p = mdb.models['Model-1'].parts['plaque']
    f = p.faces
    pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.setMeshControls(regions=pickedRegions, algorithm=ADVANCING_FRONT)
    p = mdb.models['Model-1'].parts['plaque']
    p.generateMesh()
    a1 = mdb.models['Model-1'].rootAssembly
    a1.regenerate()
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    mdb.Job(name='matrix_clean', model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
        multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
    mdb.jobs['matrix_clean'].submit(consistencyChecking=OFF)
    mdb.jobs['matrix_clean'].waitForCompletion()
    session.mdbData.summary()

    odb = session.openOdb(name='C:/temp/matrix_clean.odb')
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    session.viewports['Viewport: 1'].makeCurrent()
    nf = NumberFormat(numDigits=9, precision=0, format=ENGINEERING)
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES, 
        numberFormat=nf)
    session.writeFieldReport(fileName=report_path, append=OFF, 
        sortItem='Node Label', odb=odb, step=0, frame=0, outputPosition=NODAL, 
        variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), (
        COMPONENT, 'U3'), )), ('UR', NODAL, ((COMPONENT, 'UR1'), (COMPONENT, 
        'UR2'), (COMPONENT, 'UR3'), )), ), stepFrame=ALL)


    fname = r'c:\temp\matrix_clean.inp'
    fopen = open(fname, 'r')
    data = fopen.readlines()    
    fopen.close()

    i = 0
    for el in data:
        if '--------------------------' in el:
            #print(i)
            i_cut = i
        i = i+1

    data_mod = data[0:i_cut+1]
    add = ['*STEP\n','*MATRIX GENERATE, STIFFNESS, MASS=CONSISTENT\n','*MATRIX OUTPUT, STIFFNESS, MASS, FORMAT=MATRIX INPUT\n','*Boundary\n','_PickedSet4, ENCASTRE\n','*END STEP\n']
    for el in add:
        data_mod.append(el)

    fopen = open(r'c:\temp\matrix_clean_mod.inp', 'w')
    fopen.writelines(data_mod)
    fopen.close()

    # Lancer la génération des matrices 
    # avec abaqus job=nomdufichierinp interactive
    # récupérer les matrices MASS1 et STIF1

    import subprocess
    p = subprocess.Popen('abaqus job=matrix_clean_mod interactive', shell = True,cwd='C:/temp/')
    p.wait()

    return None

if __name__ == "__main__":
    current_path=os.getcwd()
    json_path = os.path.join(current_path, "params.json")
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    run_sim(params[0], *params[1])
    
