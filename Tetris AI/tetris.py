# Tetromino (a Tetris clone)
# By Al Sweigart al@inventwithpython.com
# Improved by Olivier Schipper
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license


import random, time, pygame, sys
from pygame.locals import *

FPS = 60
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
BOXSIZE = 20
BOARDWIDTH = 10
BOARDHEIGHT = 40
NEXTPIECES = 6
BLANK = '.'

MOVESIDEWAYSFREQ = 0.05 * FPS
MOVESIDEWAYSDELAY = 0.2 * FPS
MOVEDOWNFREQ = 0.05 * FPS
LOCKTIME = 0.5 * FPS

XMARGIN = int((WINDOWWIDTH - BOARDWIDTH * BOXSIZE) / 2)
TOPMARGIN = WINDOWHEIGHT - (BOARDHEIGHT * BOXSIZE) - 5 + 400

#               R    G    B
WHITE = (255, 255, 255)
GRAY = (185, 185, 185)
BLACK = (0, 0, 0)
RED = (155, 0, 0)
LIGHTRED = (175, 20, 20)
GREEN = (0, 155, 0)
LIGHTGREEN = (20, 175, 20)
BLUE = (20, 20, 175)
LIGHTBLUE = (40, 40, 195)
CYAN = (27, 226, 216)
LIGHTCYAN = (28, 255, 243)
YELLOW = (235, 230, 0)
LIGHTYELLOW = (255, 250, 20)
ORANGE = (234, 137, 82)
LIGHTORANGE = (255, 140, 0)
PURPLE = (136, 17, 173)
LIGHTPURPLE = (160, 30, 200)

BORDERCOLOR = BLUE
BGCOLOR = BLACK
TEXTCOLOR = WHITE
TEXTSHADOWCOLOR = GRAY
COLORS = (GREEN, RED, BLUE, ORANGE, CYAN, YELLOW, PURPLE)
LIGHTCOLORS = (LIGHTGREEN, LIGHTRED, LIGHTBLUE, LIGHTORANGE, LIGHTCYAN, LIGHTYELLOW, LIGHTPURPLE)
assert len(COLORS) == len(LIGHTCOLORS)  # each color must have light color

SHAPES = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']

S_SHAPE_TEMPLATE = [['.OO',
                     'OO.',
                     '...'],
                    ['.O.',
                     '.OO',
                     '..O'],
                    ['...',
                     '.OO',
                     'OO.'],
                    ['O..',
                     'OO.',
                     '.O.', ]]

Z_SHAPE_TEMPLATE = [['OO.',
                     '.OO',
                     '...'],
                    ['..O',
                     '.OO',
                     '.O.'],
                    ['...',
                     'OO.',
                     '.OO'],
                    ['.O.',
                     'OO.',
                     'O..']]

I_SHAPE_TEMPLATE = [['....',
                     'OOOO',
                     '....',
                     '....'],
                    ['..O.',
                     '..O.',
                     '..O.',
                     '..O.'],
                    ['....',
                     '....',
                     'OOOO',
                     '....'],
                    ['.O..',
                     '.O..',
                     '.O..',
                     '.O..']]

O_SHAPE_TEMPLATE = [['.OO.',
                     '.OO.',
                     '....'],
                    ['.OO.',
                     '.OO.',
                     '....'],
                    ['.OO.',
                     '.OO.',
                     '....'],
                    ['.OO.',
                     '.OO.',
                     '....']]

J_SHAPE_TEMPLATE = [['O..',
                     'OOO',
                     '...'],
                    ['.OO',
                     '.O.',
                     '.O.'],
                    ['OOO',
                     '..O',
                     '...'],
                    ['.O.',
                     '.O.',
                     'OO.']]

L_SHAPE_TEMPLATE = [['..O',
                     'OOO',
                     '...'],
                    ['.O.',
                     '.O.',
                     '.OO'],
                    ['...',
                     'OOO',
                     'O..'],
                    ['OO.',
                     '.O.',
                     '.O.']]

T_SHAPE_TEMPLATE = [['.O.',
                     'OOO',
                     '...'],
                    ['.O.',
                     '.OO',
                     '.O.'],
                    ['...',
                     'OOO',
                     '.O.'],
                    ['.O.',
                     'OO.',
                     '.O.']]

T_SPIN_CHECK_TEMPLATE = ['O.O',
                         '...',
                         'O.O']

PIECES = {'S': S_SHAPE_TEMPLATE,
          'Z': Z_SHAPE_TEMPLATE,
          'J': J_SHAPE_TEMPLATE,
          'L': L_SHAPE_TEMPLATE,
          'I': I_SHAPE_TEMPLATE,
          'O': O_SHAPE_TEMPLATE,
          'T': T_SHAPE_TEMPLATE}

PIECES_COLORS = {'S': 0,
                 'Z': 1,
                 'J': 2,
                 'L': 3,
                 'I': 4,
                 'O': 5,
                 'T': 6}

WALL_KICK_DATA = {'01': [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
                  '10': [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
                  '12': [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
                  '21': [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
                  '23': [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
                  '32': [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
                  '30': [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
                  '03': [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]}

WALL_KICK_DATA_I = {'01': [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
                    '10': [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
                    '12': [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
                    '21': [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
                    '23': [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
                    '32': [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
                    '30': [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
                    '03': [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)]}

SCORING_DATA = {'0': 0,
                'B0': 0,
                '1': 100,
                'B1': 100,
                'MT0': 100,
                'MT1': 200,
                'BMT1': 200,
                '2': 300,
                'B2': 300,
                'T0': 400,
                'BT0': 400,
                '3': 500,
                'B3': 500,
                '4': 800,
                'T1': 800,
                'BMT0': 150,
                'BT1': 1200,
                'B4': 1200,
                'T2': 1200,
                'T3': 1600,
                'BT2': 1800,
                'BT3': 2400}

LINE_CLEAR_DATA = {'0': 0,
                   '1': 1,
                   '2': 3,
                   'T0': 1,
                   '3': 5,
                   '4': 8,
                   'B4': 12,
                   'T1': 3,
                   'T2': 7,
                   'T3': 6}

pygame.mixer.pre_init(48000, 16, 2, 4096)
pygame.init()
EFFECT_MOVE = pygame.mixer.Sound('move.wav')
EFFECT_ROTATE = pygame.mixer.Sound('rotate.wav')
EFFECT_HOLD = pygame.mixer.Sound('hold.wav')
EFFECT_LOCK = pygame.mixer.Sound('lock.wav')
EFFECT_CLEAR = pygame.mixer.Sound('clear.wav')

class Bag:
    def __init__(self):
        self.bag = []

    def newBag(self):
        self.bag = [n for n in range(0, len(PIECES.keys()))]
        random.shuffle(self.bag)

    def getPiece(self):
        if len(self.bag) == 0:
            self.newBag()
        index = random.choice(self.bag)
        self.bag.remove(index)
        return SHAPES[index]


def main():
    global DISPLAYSURF, BASICFONT, BIGFONT, BAG
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
    pygame.display.set_caption('Tetromino')
    BAG = Bag()

    showTextScreen('Tetromino')
    while True:  # game loop
        if random.randint(0, 1) == 0:
            pygame.mixer.music.load('tetrisb.mid')
        else:
            pygame.mixer.music.load('tetrisc.mid')
        pygame.mixer.music.play(-1, 0.0)
        runGame()
        pygame.mixer.music.stop()
        showTextScreen('Game Over')


def runGame():
    # setup variables for the start of the game
    frame = 0
    board = getBlankBoard()
    BAG.newBag()
    lastMoveDownTime = frame
    lastMoveSidewaysTime = frame
    lastMoveSidewaysInput = frame
    lastFallTime = 0
    movingDown = False  # note: there is no movingUp variable
    movingLeft = False
    movingRight = False
    canUseHold = True
    lockTime = LOCKTIME
    lastPieceLock = None
    lastSuccessfulMovement = None
    ghostPieceYOffset = 0
    score = 0
    combo = 0
    lines = 0
    moves = 0
    linesGoal = 5
    level = 1
    level, fallFreq, linesGoal = calculateLevelAndFallFreq(lines, linesGoal, level)

    fallingPiece = getNewPiece()
    nextPieces = [getNewPiece() for _ in range(NEXTPIECES)]
    holdPiece = None

    while True:  # game loop
        if fallingPiece is None:
            # No falling piece in play, so start a new piece at the top
            fallingPiece = nextPieces.pop(0)
            nextPieces.append(getNewPiece())
            lastFallTime = 0  # reset lastFallTime
            lockTime = LOCKTIME
            moves = 0
            canUseHold = True

            if not isValidPosition(board, fallingPiece):
                return  # can't fit a new piece on the board, so game over

        checkForQuit()
        for event in pygame.event.get():  # event handling loop
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                    # Pausing the game
                    DISPLAYSURF.fill(BGCOLOR)
                    pygame.mixer.music.stop()
                    showTextScreen('Paused')  # pause until a key press
                    pygame.mixer.music.play(-1, 0.0)
                    lastFallTime = frame
                    lastMoveDownTime = frame
                    lastMoveSidewaysTime = frame
                elif event.key == K_LEFT or event.key == K_a:
                    movingLeft = False
                elif event.key == K_RIGHT or event.key == K_d:
                    movingRight = False
                elif event.key == K_DOWN or event.key == K_s:
                    movingDown = False

            elif event.type == KEYDOWN:
                # moving the piece sideways
                if (event.key == K_LEFT or event.key == K_a) and isValidPosition(board, fallingPiece, adjX=-1):
                    fallingPiece['x'] -= 1
                    movingLeft = True
                    movingRight = False
                    lastMoveSidewaysTime = frame
                    lastMoveSidewaysInput = frame
                    lastSuccessfulMovement = "moveLeft"
                    EFFECT_MOVE.play()
                    if not lockTime == LOCKTIME and moves < 16:
                        lockTime = LOCKTIME
                        moves += 1

                elif (event.key == K_RIGHT or event.key == K_d) and isValidPosition(board, fallingPiece, adjX=1):
                    fallingPiece['x'] += 1
                    movingRight = True
                    movingLeft = False
                    lastMoveSidewaysTime = frame
                    lastMoveSidewaysInput = frame
                    lastSuccessfulMovement = "moveRight"
                    EFFECT_MOVE.play()
                    if not lockTime == LOCKTIME and moves < 16:
                        lockTime = LOCKTIME
                        moves += 1

                # rotating the piece (if there is room to rotate)
                elif event.key == K_UP or event.key == K_x:
                    if rotatePiece(fallingPiece, 1, board):
                        lastSuccessfulMovement = "rotate"
                        EFFECT_ROTATE.play()
                        if not lockTime == LOCKTIME and moves < 16:
                            lockTime = LOCKTIME
                            moves += 1
                elif event.key == K_z:  # rotate the other direction
                    if rotatePiece(fallingPiece, -1, board):
                        lastSuccessfulMovement = "rotate"
                        EFFECT_ROTATE.play()
                        if not lockTime == LOCKTIME and moves < 16:
                            lockTime = LOCKTIME
                            moves += 1

                # making the piece fall faster with the down key
                elif event.key == K_DOWN or event.key == K_s:
                    movingDown = True
                    if isValidPosition(board, fallingPiece, adjY=1):
                        fallingPiece['y'] += 1
                        score += 1
                        lastSuccessfulMovement = "moveDown"
                        lastMoveDownTime = frame

                # move the current piece all the way down
                elif event.key == K_SPACE:
                    movingDown = False
                    movingLeft = False
                    movingRight = False
                    lockTime = -1
                    for i in range(1, BOARDHEIGHT):
                        if not isValidPosition(board, fallingPiece, adjY=i):
                            fallingPiece['y'] += i - 1
                            score += 2 * (i - 1)
                            lastSuccessfulMovement = "moveDown"
                            break

                # hold piece
                elif (event.key == K_c) and canUseHold:
                    # swap hold and falling piece
                    EFFECT_HOLD.play()
                    oldHoldPiece = holdPiece
                    holdPiece = fallingPiece
                    fallingPiece = oldHoldPiece
                    if fallingPiece is None:
                        fallingPiece = nextPieces.pop(0)
                        nextPieces.append(getNewPiece())

                    lastFallTime = 0  # reset lastFallTime
                    lockTime = LOCKTIME
                    moves = 0

                    holdPiece['rotation'] = 0
                    holdPiece['x'] = 3
                    holdPiece['y'] = 18
                    canUseHold = False

        # handle moving the piece because of user input
        if (
                movingLeft or movingRight) and frame - lastMoveSidewaysTime > MOVESIDEWAYSFREQ and frame - lastMoveSidewaysInput > MOVESIDEWAYSDELAY:
            if movingLeft and isValidPosition(board, fallingPiece, adjX=-1):
                fallingPiece['x'] -= 1
                lastSuccessfulMovement = "moveLeft"
                EFFECT_MOVE.play()
            elif movingRight and isValidPosition(board, fallingPiece, adjX=1):
                fallingPiece['x'] += 1
                lastSuccessfulMovement = "moveRight"
                EFFECT_MOVE.play()
            lastMoveSidewaysTime = frame

        if movingDown and frame - lastMoveDownTime > MOVEDOWNFREQ:
            if isValidPosition(board, fallingPiece, adjY=1):
                fallingPiece['y'] += 1
                score += 1
                lastSuccessfulMovement = "moveDown"
                lastMoveDownTime = frame

        if not isValidPosition(board, fallingPiece, adjY=1):
            lastFallTime = frame
            lockTime -= 1
            if lockTime < 0:
                if fallingPiece['y'] < 19:
                    return  # lock out: a piece locked in above the screen

                # check for T-spin
                tspin = checkForTSpin(fallingPiece, board, lastSuccessfulMovement)

                addToBoard(board, fallingPiece)
                linesCleared = removeCompleteLines(board)

                # line values for variable-goal levels
                if linesCleared == 0:
                    combo = 0
                    EFFECT_LOCK.play()
                elif linesCleared == 1:
                    score += 20 * combo * level
                    combo += 1
                    EFFECT_CLEAR.play()
                else:
                    score += 50 * combo * level
                    combo += 1
                    EFFECT_CLEAR.play()

                currentAction = str(linesCleared)
                currentAction = tspin + currentAction

                if currentAction == lastPieceLock:
                    currentAction = "B" + currentAction

                try:
                    lines += LINE_CLEAR_DATA[currentAction]
                except KeyError:
                    pass

                lastPieceLock = currentAction
                score += SCORING_DATA[currentAction] * level

                level, fallFreq, linesGoal = calculateLevelAndFallFreq(lines, linesGoal, level)
                fallingPiece = None

        # let the piece fall if it is time to fall
        if frame - lastFallTime > fallFreq:
            lastFallTime = frame
            # see if the piece has landed
            if isValidPosition(board, fallingPiece, adjY=1):
                fallingPiece['y'] += 1
                lastSuccessfulMovement = "moveDown"

        # calculate the ghost piece
        if fallingPiece is not None:
            for i in range(1, BOARDHEIGHT):
                if not isValidPosition(board, fallingPiece, adjY=i):
                    ghostPieceYOffset = i - 1
                    break

        # drawing everything on the screen
        DISPLAYSURF.fill(BGCOLOR)
        drawStatus(score, level)
        drawNextPieces(nextPieces)
        drawHoldPiece(holdPiece)
        if fallingPiece is not None:
            drawGhostPiece(fallingPiece, ghostPieceYOffset)
            drawPiece(fallingPiece)
        drawBoard(board)
        drawBoxToObscurePiece()

        frame += 1
        pygame.display.update()


def checkForTSpin(piece, board, lastSuccessfulMovement):
    # check if T
    if piece['shape'] != 'T':
        return ""

    # check if the last succesfull movement was a rotation
    if lastSuccessfulMovement != "rotate":
        return ""

    # return "T" if 3 or more diagonals are obstructed
    diagonalsObstructed = 0
    for x in range(getWidth(piece)):
        for y in range(getHeight(piece)):
            if T_SPIN_CHECK_TEMPLATE[y][x] == BLANK:
                continue
            if board[x + piece['x']][y + piece['y']] != BLANK:
                diagonalsObstructed += 1

    if diagonalsObstructed >= 3:
        return "T"
    else:
        return ""


def rotatePiece(piece, rotation, board):
    initialRotation = piece['rotation']
    desiredRotation = (piece['rotation'] + rotation) % len(PIECES[piece['shape']])
    if piece['shape'] == 'I':
        for x, y in WALL_KICK_DATA_I[str(initialRotation) + str(desiredRotation)]:
            piece['rotation'] = desiredRotation
            piece['x'] = piece['x'] + x
            piece['y'] = piece['y'] - y
            if isValidPosition(board, piece):
                return True
            else:
                piece['rotation'] = initialRotation
                piece['x'] = piece['x'] - x
                piece['y'] = piece['y'] + y
    elif piece['shape'] == 'O':
        return False
    else:
        for x, y in WALL_KICK_DATA[str(initialRotation) + str(desiredRotation)]:
            piece['rotation'] = desiredRotation
            piece['x'] = piece['x'] + x
            piece['y'] = piece['y'] - y
            if isValidPosition(board, piece):
                return True
            else:
                piece['rotation'] = initialRotation
                piece['x'] = piece['x'] - x
                piece['y'] = piece['y'] + y


def makeTextObjs(text, font, draw_color):
    surf = font.render(text, True, draw_color)
    return surf, surf.get_rect()


def terminate():
    pygame.quit()
    sys.exit()


def checkForKeyPress():
    # Go through event queue looking for a KEYUP event.
    # Grab KEYDOWN events to remove them from the event queue.
    checkForQuit()

    for event in pygame.event.get([KEYDOWN, KEYUP]):
        if event.type == KEYDOWN:
            continue
        return event.key
    return None


def showTextScreen(text):
    # This function displays large text in the
    # center of the screen until a key is pressed.
    # Draw the text drop shadow
    titleSurf, titleRect = makeTextObjs(text, BIGFONT, TEXTSHADOWCOLOR)
    titleRect.center = (int(WINDOWWIDTH / 2), int(WINDOWHEIGHT / 2))
    DISPLAYSURF.blit(titleSurf, titleRect)

    # Draw the text
    titleSurf, titleRect = makeTextObjs(text, BIGFONT, TEXTCOLOR)
    titleRect.center = (int(WINDOWWIDTH / 2) - 3, int(WINDOWHEIGHT / 2) - 3)
    DISPLAYSURF.blit(titleSurf, titleRect)

    # Draw the additional "Press a key to play." text.
    pressKeySurf, pressKeyRect = makeTextObjs('Press a key to play.', BASICFONT, TEXTCOLOR)
    pressKeyRect.center = (int(WINDOWWIDTH / 2), int(WINDOWHEIGHT / 2) + 100)
    DISPLAYSURF.blit(pressKeySurf, pressKeyRect)

    while checkForKeyPress() is None:
        pygame.display.update()
        FPSCLOCK.tick()


def checkForQuit():
    for _ in pygame.event.get(QUIT):  # get all the QUIT events
        terminate()  # terminate if any QUIT events are present
    # for event in pygame.event.get(KEYUP):  # get all the KEYUP events
    #     if event.key == K_ESCAPE:
    #         terminate()  # terminate if the KEYUP event was for the Esc key
    #     pygame.event.post(event)  # put the other KEYUP event objects back


def calculateLevelAndFallFreq(lines, linesGoal, level):
    # Based on the score, return the level the player is on and
    # how many seconds pass until a falling piece falls one space.
    if lines >= linesGoal:
        level += 1
        linesGoal += 5 * level

    fallFreq = ((0.8 - ((level - 1) * 0.007)) ** (level - 1)) * FPS
    return level, fallFreq, linesGoal


def getNewPiece():
    # return a random new piece in a random rotation and color
    shape = BAG.getPiece()
    newPiece = {'shape': shape,
                'rotation': 0,
                'x': 3,
                'y': 18,  # start it above the board (i.e. 18 because board is 40 high)
                'color': PIECES_COLORS[shape]}
    return newPiece


def addToBoard(board, piece):
    # fill in the board based on piece's location, shape, and rotation
    for x in range(getWidth(piece)):
        for y in range(getHeight(piece)):
            if PIECES[piece['shape']][piece['rotation']][y][x] != BLANK:
                board[x + piece['x']][y + piece['y']] = piece['color']


def getBlankBoard():
    # create and return a new blank board data structure
    board = []
    for i in range(BOARDWIDTH):
        board.append([BLANK] * BOARDHEIGHT)
    return board


def isOnBoard(x, y):
    return 0 <= x < BOARDWIDTH and y < BOARDHEIGHT


def isValidPosition(board, piece, adjX=0, adjY=0):
    # Return True if the piece is within the board and not colliding
    for x in range(getWidth(piece)):
        for y in range(getHeight(piece)):
            isAboveBoard = y + piece['y'] + adjY < 0
            if isAboveBoard or PIECES[piece['shape']][piece['rotation']][y][x] == BLANK:
                continue
            if not isOnBoard(x + piece['x'] + adjX, y + piece['y'] + adjY):
                return False
            if board[x + piece['x'] + adjX][y + piece['y'] + adjY] != BLANK:
                return False
    return True


def isCompleteLine(board, y):
    # Return True if the line filled with boxes with no gaps.
    for x in range(BOARDWIDTH):
        if board[x][y] == BLANK:
            return False
    return True


def removeCompleteLines(board):
    # Remove any completed lines on the board, move everything above them down, and return the number of complete lines.
    numLinesRemoved = 0
    y = BOARDHEIGHT - 1  # start y at the bottom of the board
    while y >= 0:
        if isCompleteLine(board, y):
            # Remove the line and pull boxes down by one line.
            for pullDownY in range(y, 0, -1):
                for x in range(BOARDWIDTH):
                    board[x][pullDownY] = board[x][pullDownY - 1]
            # Set very top line to blank.
            for x in range(BOARDWIDTH):
                board[x][0] = BLANK
            numLinesRemoved += 1
            # Note on the next iteration of the loop, y is the same.
            # This is so that if the line that was pulled down is also
            # complete, it will be removed.
        else:
            y -= 1  # move on to check next row up
    return numLinesRemoved


def convertToPixelCoords(boxx, boxy):
    # Convert the given xy coordinates of the board to xy
    # coordinates of the location on the screen.
    return (XMARGIN + (boxx * BOXSIZE)), (TOPMARGIN - 400 + (boxy * BOXSIZE))


def drawBox(boxx, boxy, draw_color, pixelx=None, pixely=None, ghost=False):
    # draw a single box (each tetromino piece has four boxes)
    # at xy coordinates on the board. Or, if pixelx & pixely
    # are specified, draw to the pixel coordinates stored in
    # pixelx & pixely (this is used for the "Next" piece).
    if draw_color == BLANK:
        return
    if pixelx is None and pixely is None:
        pixelx, pixely = convertToPixelCoords(boxx, boxy)
    pygame.draw.rect(DISPLAYSURF, COLORS[draw_color], (pixelx + 1, pixely + 1, BOXSIZE - 1, BOXSIZE - 1))
    if ghost:
        pygame.draw.rect(DISPLAYSURF, [0, 0, 0], (pixelx + 3, pixely + 3, BOXSIZE - 5, BOXSIZE - 5))
    else:
        pygame.draw.rect(DISPLAYSURF, LIGHTCOLORS[draw_color], (pixelx + 1, pixely + 1, BOXSIZE - 4, BOXSIZE - 4))


def drawBoard(board):
    # fill the background of the board
    # pygame.draw.rect(DISPLAYSURF, BGCOLOR, (XMARGIN, TOPMARGIN, BOXSIZE * BOARDWIDTH, BOXSIZE * BOARDHEIGHT))
    # draw the individual boxes on the board
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            drawBox(x, y, board[x][y])

    # draw the border around the board
    pygame.draw.rect(DISPLAYSURF, BORDERCOLOR,
                     (XMARGIN - 3, TOPMARGIN - 7, (BOARDWIDTH * BOXSIZE) + 8, (BOARDHEIGHT * BOXSIZE) + 8), 5)


def drawBoxToObscurePiece():
    pygame.draw.rect(DISPLAYSURF, BGCOLOR, (XMARGIN, TOPMARGIN - 10, BOXSIZE * BOARDWIDTH, -100))


def drawStatus(score, level):
    # draw the score text
    scoreSurf = BASICFONT.render('Score: %s' % score, True, TEXTCOLOR)
    scoreRect = scoreSurf.get_rect()
    scoreRect.topleft = (WINDOWWIDTH - 150, 20)
    DISPLAYSURF.blit(scoreSurf, scoreRect)

    # draw the level text
    levelSurf = BASICFONT.render('Level: %s' % level, True, TEXTCOLOR)
    levelRect = levelSurf.get_rect()
    levelRect.topleft = (WINDOWWIDTH - 150, 50)
    DISPLAYSURF.blit(levelSurf, levelRect)


def getWidth(piece):
    return len(PIECES[piece['shape']][0][0])


def getHeight(piece):
    return len(PIECES[piece['shape']][0])


def drawPiece(piece, pixelx=None, pixely=None):
    shapeToDraw = PIECES[piece['shape']][piece['rotation']]
    if pixelx is None and pixely is None:
        # if pixelx & pixely hasn't been specified, use the location stored in the piece data structure
        pixelx, pixely = convertToPixelCoords(piece['x'], piece['y'])

    # draw each of the boxes that make up the piece
    for x in range(getWidth(piece)):
        for y in range(getHeight(piece)):
            if shapeToDraw[y][x] != BLANK:
                drawBox(None, None, piece['color'], pixelx + (x * BOXSIZE), pixely + (y * BOXSIZE))


def drawGhostPiece(piece, ghostPieceYOffset):
    shapeToDraw = PIECES[piece['shape']][piece['rotation']]
    pixelx, pixely = convertToPixelCoords(piece['x'], piece['y'] + ghostPieceYOffset)

    # draw each of the boxes that make up the piece
    for x in range(getWidth(piece)):
        for y in range(getHeight(piece)):
            if shapeToDraw[y][x] != BLANK:
                drawBox(None, None, piece['color'], pixelx + (x * BOXSIZE), pixely + (y * BOXSIZE), ghost=True)


def drawNextPieces(pieces):
    # draw the "next" text
    nextSurf = BASICFONT.render('Next:', True, TEXTCOLOR)
    nextRect = nextSurf.get_rect()
    nextRect.topleft = (WINDOWWIDTH - 120, 80)
    DISPLAYSURF.blit(nextSurf, nextRect)
    # draw the "next" pieces
    for i, p in enumerate(pieces):
        drawPiece(p, pixelx=WINDOWWIDTH - 120, pixely=100 + 45 * i)


def drawHoldPiece(piece):
    # draw the "hold" text
    holdSurf = BASICFONT.render('Hold:', True, TEXTCOLOR)
    holdRect = holdSurf.get_rect()
    holdRect.topleft = (WINDOWWIDTH - 520, 80)
    DISPLAYSURF.blit(holdSurf, holdRect)
    # draw the "hold" piece
    if piece is not None:
        drawPiece(piece, pixelx=WINDOWWIDTH - 520, pixely=100)


if __name__ == '__main__':
    main()
