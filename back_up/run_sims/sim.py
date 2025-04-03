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
            largeur = 405.0, #mm
            hauteur = 403.0, #mm
            distance = 25.0, #mm
            rayon = 4.5, #mm
            plaque_epaisseur =  1.0, #mm
            frequ_max_mode = 500.0,
            #mesh params
            elem_size = 20,
            deviationFactor=0.1,
            minSizeFactor=0.1
            ) : #vérifier unité si Hz ou rad/s?


    #resultat dans le fichier .odb
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
        sheetSize=500.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Spot(point=(0.0, 0.0))
    s.FixedConstraint(entity=v[0])
    #geometrie de la plaque
    s.rectangle(point1=(0.0, 0.0), point2=(largeur, hauteur))

    # definition des trous de fixation - arrête du trou est bloquée
    s.CircleByCenterPerimeter(center=(distance, distance), point1=(distance, distance + rayon))
    s.CircleByCenterPerimeter(center=(distance, hauteur - distance), point1=(distance, hauteur - distance - rayon))
    s.CircleByCenterPerimeter(center=(largeur - distance, hauteur - distance), point1=(largeur - distance, hauteur - distance - rayon))
    s.CircleByCenterPerimeter(center=(largeur - distance, distance), point1=(largeur - distance, distance + rayon))


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
    mdb.models['Model-1'].Material(name='acier')
    mdb.models['Model-1'].materials['acier'].Density(table=((7.86e-09, ), ))
    mdb.models['Model-1'].materials['acier'].Elastic(table=((210000.0, 0.26), ))
    # thickness est l'épaisseru de la pièces en mm
    mdb.models['Model-1'].HomogeneousShellSection(name='tole', preIntegrate=OFF, 
        material='acier', thicknessType=UNIFORM, thickness = plaque_epaisseur, 
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
    # frequence propre max en Hz
    mdb.models['Model-1'].FrequencyStep(name='analyse_modale', previous='Initial', 
        maxEigen=frequ_max_mode )
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        step='analyse_modale')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
        constraints=ON, connectors=ON, engineeringFeatures=ON, 
        adaptiveMeshConstraints=OFF)
    a = mdb.models['Model-1'].rootAssembly
    e1 = a.instances['plaque-1'].edges
    a.ReferencePoint(point=a.instances['plaque-1'].InterestingPoint(edge=e1[2], 
        rule=CENTER))
    a = mdb.models['Model-1'].rootAssembly
    e11 = a.instances['plaque-1'].edges
    a.ReferencePoint(point=a.instances['plaque-1'].InterestingPoint(edge=e11[1], 
        rule=CENTER))
    a = mdb.models['Model-1'].rootAssembly
    e1 = a.instances['plaque-1'].edges
    a.ReferencePoint(point=a.instances['plaque-1'].InterestingPoint(edge=e1[0], 
        rule=CENTER))
    a = mdb.models['Model-1'].rootAssembly
    e11 = a.instances['plaque-1'].edges
    a.ReferencePoint(point=a.instances['plaque-1'].InterestingPoint(edge=e11[3], 
        rule=CENTER))
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.referencePoints
    refPoints1=(r1[4], )
    region1=regionToolset.Region(referencePoints=refPoints1)
    a = mdb.models['Model-1'].rootAssembly
    s1 = a.instances['plaque-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#4 ]', ), )
    region2=regionToolset.Region(side1Edges=side1Edges1)
    mdb.models['Model-1'].Coupling(name='Constraint-1', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.referencePoints
    refPoints1=(r1[5], )
    region1=regionToolset.Region(referencePoints=refPoints1)
    a = mdb.models['Model-1'].rootAssembly
    s1 = a.instances['plaque-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#2 ]', ), )
    region2=regionToolset.Region(side1Edges=side1Edges1)
    mdb.models['Model-1'].Coupling(name='Constraint-2', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.referencePoints
    refPoints1=(r1[6], )
    region1=regionToolset.Region(referencePoints=refPoints1)
    a = mdb.models['Model-1'].rootAssembly
    s1 = a.instances['plaque-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#1 ]', ), )
    region2=regionToolset.Region(side1Edges=side1Edges1)
    mdb.models['Model-1'].Coupling(name='Constraint-3', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.referencePoints
    refPoints1=(r1[7], )
    region1=regionToolset.Region(referencePoints=refPoints1)
    a = mdb.models['Model-1'].rootAssembly
    s1 = a.instances['plaque-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#8 ]', ), )
    region2=regionToolset.Region(side1Edges=side1Edges1)
    mdb.models['Model-1'].Coupling(name='Constraint-4', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, interactions=OFF, constraints=OFF, 
        engineeringFeatures=OFF)
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.referencePoints
    refPoints1=(r1[4], r1[5], r1[6], r1[7], )
    region = regionToolset.Region(referencePoints=refPoints1)
    mdb.models['Model-1'].EncastreBC(name='blocage', 
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
    # définir le calcul
    
    mdb.Job(name='vibration', model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, numDomains=1,
        multiprocessingMode=DEFAULT, numCpus=1, numGPUs=1)
    # lance le calcul
    mdb.jobs['vibration'].submit(consistencyChecking=OFF)
    #mdb.jobs['vibration'].waitForCompletion()
    session.mdbData.summary()
    o3 = session.openOdb(
        name='C:/temp/vibration.odb')
    session.viewports['Viewport: 1'].setValues(displayedObject=o3)
    session.viewports['Viewport: 1'].makeCurrent()
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
        CONTOURS_ON_DEF, ))
    # export to FWF
    odb = session.odbs['C:/temp/vibration.odb']
    session.fieldReportOptions.setValues(printTotal=OFF, printMinMax=OFF)
    session.writeFieldReport(fileName=report_path, append=OFF, sortItem='Node Label', odb=odb, step=0, frame=0, outputPosition=NODAL, variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), (COMPONENT, 'U3'), )), ), stepFrame=ALL)
    #session.writeFieldReport(fileName='abaqus.rpt', append=OFF, sortItem='Node Label', odb=odb, step=0, frame=29, outputPosition=NODAL, variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), (COMPONENT, 'U3'), )), ), stepFrame=ALL)



    return None

if __name__ == "__main__":
    current_path=os.getcwd()
    json_path = os.path.join(current_path, "params.json")
    with open(json_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    run_sim(params[0], *params[1])
    
