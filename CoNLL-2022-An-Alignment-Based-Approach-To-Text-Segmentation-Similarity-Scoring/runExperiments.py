from functools import reduce
from turtle import right
from helpers import sgmGenerator, writeLinesToFile, massLen, massToBinStr, sgmPairGenerator, isSoftTransp
from nltk.metrics.segmentation import windowdiff
import segeval
from similarityMetrics import alignmentIndex


def isEquidistant(gold,h1,h2):

    def _getWDPaddedSgm(massList,k):
        paddedMass = massList[:]
        paddedMass[0]+= k-1
        paddedMass[-1]+= k-1

        return paddedMass
    

    k = max(1,round(reduce(lambda x,y: x+y, gold)/len(gold)/2))

    g1w = _getWDPaddedSgm(gold,k)
    h1w = _getWDPaddedSgm(h1,k)
    h2w = _getWDPaddedSgm(h2,k)

    wd1 = windowdiff(massToBinStr(g1w),massToBinStr(h1w),k)
    wd2 = windowdiff(massToBinStr(g1w),massToBinStr(h2w),k)
    
    b1 = segeval.boundary_similarity(gold,h1)
    b2 = segeval.boundary_similarity(gold,h2)

    z1 = alignmentIndex(gold,h1)
    z2 = alignmentIndex(gold,h2)

    return (wd1==wd2),(b1==b2),(z1==z2)

def getShiftJaccard(leftSize,rightSize,shift):
    if shift>0 and shift>=rightSize:
        raise ValueError(f'Positive Offset is >= than Right Segment. Offset:{shift} RightSize: {rightSize}')
    if shift<0 and abs(shift)>=leftSize:
        raise ValueError(f'Negative Offset is >= than Left Segment. Offset:{shift} LeftSize: {leftSize}')
    
    if shift==0:
        return 1,1
    
    dist = abs(shift)
    j1 = (leftSize-dist)/(leftSize) if shift < 0 else (leftSize)/(leftSize+dist)
    j2 = (rightSize)/(rightSize+dist) if shift < 0 else (rightSize-dist)/(rightSize)

    return j1, j2

def isSoftTransp(leftSize,rightSize,shift):
    j1, j2 = getShiftJaccard(leftSize,rightSize,shift)
    return j1>0.5 and j2>0.5


def hasConstCostTranspErr(gold):

    wdConfuses = bConfuses = zConfuses = False

    #for every possible boundary
    for i in range(len(gold)-1):
        l1 = gold[i]
        r1 = gold[i+1]


        offsetsA = [x for x in range(-(l1-1),(r1-1)+1) if x != 0]
        
        #check all possible transpositions for that boundary
        for shiftA in offsetsA:

            j1, j2 = getShiftJaccard(l1,r1,shiftA)

            isSoftA =  j1>0.5 and j2>0.5
            isHardA =  j1<0.5 and j2<0.5

            h1 = gold[:i] + [l1+shiftA,r1-shiftA] + gold[i+2:]

            shiftSize = abs(shiftA)

            #check all boundaries and see if a transposition of the same distance is available with different impact
            for j in range(i,len(gold)-1):
                l2 = gold[j]
                r2 = gold[j+1]

                offsetsB = []
                if l2>shiftSize:
                    offsetsB.append(-shiftSize)
                if r2>shiftSize:
                    offsetsB.append(shiftSize)

                for shiftB in offsetsB:
                    if i==j and shiftA==shiftB:
                        continue

                    j3, j4 = getShiftJaccard(l2,r2,shiftB)

                    isSoftB = j3>0.5 and j4>0.5
                    isHardB = j3<0.5 and j4<0.5

                    if (isSoftA and isHardB) or (isHardA and isSoftB):

                        h2 = gold[:j] + [l2+shiftB,r2-shiftB] + gold[j+2:]

                        wd, b, z = isEquidistant(gold,h1,h2)

                        wdConfuses = wdConfuses or wd
                        bConfuses = bConfuses or b
                        zConfuses = zConfuses or z

                        if wdConfuses and bConfuses and zConfuses:
                            return True,True,True

    return wdConfuses, bConfuses, zConfuses

def hasCrossTranspErr(gold):

    cache = {}
    def getSoftTransps(dist):
        if dist not in cache:
            t = []
            for i in range(len(gold)-1):
                leftSize = gold[i]
                rightSize = gold[i+1]

                if leftSize>dist:
                    j1, j2 = getShiftJaccard(leftSize,rightSize,-dist)
                    if j1>0.5 and j2>0.5:
                        h = gold[:i] + [leftSize-dist,rightSize+dist] + gold[i+2:]
                        t.append(h)

                if rightSize>dist:
                    j1, j2 = getShiftJaccard(leftSize,rightSize,dist)
                    if j1>0.5 and j2>0.5:
                        h = gold[:i] + [leftSize+dist,rightSize-dist] + gold[i+2:]
                        t.append(h)
            
            cache[dist] = t
        
        return cache[dist]

    wdConfuses = bConfuses = zConfuses = False

    for i in range(len(gold)-2):
        left = gold[i]
        middle = gold[i+1]
        right = gold[i+2]

        leftBoundCrossDist = [x for x in range (1,right)]
        rightBoundCrossDist = [x for x in range (1,left)]

        for crossDist in leftBoundCrossDist:
            transpDist = middle + crossDist

            h2 = gold[:i] + [left+middle,crossDist,right-crossDist] + gold[i+3:]
            hStar = [h for h in getSoftTransps(transpDist) if h!=h2]
            for h1 in hStar:
                wd, b, z = isEquidistant(gold,h1,h2)

                wdConfuses = wdConfuses or wd
                bConfuses = bConfuses or b
                zConfuses = zConfuses or z

                if wdConfuses and bConfuses and zConfuses:
                    return True,True,True

        for crossDist in rightBoundCrossDist:
            transpDist = middle + crossDist
            h2 = gold[:i] + [left-crossDist,crossDist,right+middle] + gold[i+3:]
            hStar = [h for h in getSoftTransps(transpDist) if h!=h2]
            for h1 in hStar:
                wd, b, z = isEquidistant(gold,h1,h2)

                wdConfuses = wdConfuses or wd
                bConfuses = bConfuses or b
                zConfuses = zConfuses or z

                if wdConfuses and bConfuses and zConfuses:
                    return True,True,True

    return wdConfuses,bConfuses,zConfuses

def hasVanishTranspErr(gold):

    wdConfuses = bConfuses = zConfuses = False

    for i in range(len(gold)-1):
        leftSize = gold[i]
        rightSize = gold[i+1]

        #get all the transposition offsets
        offsets = [x for x in range(-(leftSize-1),(rightSize-1)+1) if x!=0]
        
        #compare every pair of transposition offsets (on same boundary)
        for j in range(len(offsets)):

            shiftA = offsets[j]

            for k in range(j+1,len(offsets)):
                shiftB = offsets[k]

                if abs(shiftA)==abs(shiftB):
                    continue

                assert abs(shiftA)!=abs(shiftB)

                smallShift = bigShift = None
                if abs(shiftA)<abs(shiftB):
                    smallShift = shiftA
                    bigShift = shiftB
                elif abs(shiftB)<abs(shiftA):
                    smallShift = shiftB
                    bigShift = shiftA

                #we only want pairs of transpositions where the 'small' transposition is soft
                if not isSoftTransp(leftSize,rightSize,smallShift):
                    continue

                #check if the pair is 'confounded'
                h1 = gold[:i] + [leftSize+smallShift,rightSize-smallShift] + gold[i+2:]
                h2 = gold[:i] + [leftSize+bigShift,rightSize-bigShift] + gold[i+2:]

                wd, b, z = isEquidistant(gold,h1,h2)

                wdConfuses = wdConfuses or wd
                bConfuses = bConfuses or b
                zConfuses = zConfuses or z

                if wdConfuses and bConfuses and zConfuses:
                    return True,True,True
        
    return wdConfuses, bConfuses, zConfuses


def runExp(name,maxInstanceLen):

    tests = {
        'constCostTransp' : hasConstCostTranspErr,
        'crossTransp': hasCrossTranspErr,
        'vanishTransp': hasVanishTranspErr
    }


    lines = ['length,nSegments,instanceSpace,wdCount,bCount,zCount']
    for size in range(5,maxInstanceLen+1):
        nSegments = 1
        for sgmSet in sgmGenerator(size):
            nSegments +=1
            wdConfuses = 0
            bConfuses = 0
            zConfuses = 0
            n = 0
            for instance in sgmSet:
                n += 1
                w, b , z = tests[name](instance)
                wdConfuses += w
                bConfuses += b
                zConfuses += z
                

            lines.append(f"{size},{nSegments},{n},{wdConfuses},{bConfuses},{zConfuses}")
        print(size)

    outputFile = f'./Results/{name}-Results.csv'
    writeLinesToFile(lines,outputFile)

if __name__ == "__main__":
    maxLen = 20
    runExp('constCostTransp',maxLen)
    runExp('crossTransp',maxLen)
    runExp('vanishTransp',maxLen)



            








