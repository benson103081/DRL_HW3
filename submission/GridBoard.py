import numpy as np
import random


def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


class BoardPiece:
    def __init__(self, name, code, pos):
        self.name = name
        self.code = code
        self.pos = pos


class BoardMask:
    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self):
        return np.nonzero(self.mask)


def zip_positions2d(positions):
    x, y = positions
    return list(zip(x, y))


class GridBoard:
    def __init__(self, size=4):
        self.size = size
        self.components = {}
        self.masks = {}

    def addPiece(self, name, code, pos=(0, 0)):
        self.components[name] = BoardPiece(name, code, pos)

    def addMask(self, name, mask, code):
        self.masks[name] = BoardMask(name, mask, code)

    def movePiece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        if move:
            self.components[name].pos = pos

    def render(self):
        displ_board = np.zeros((self.size, self.size), dtype='<U2')
        displ_board[:] = ' '
        for name, piece in self.components.items():
            displ_board[piece.pos] = piece.code
        for name, mask in self.masks.items():
            displ_board[mask.get_positions()] = mask.code
        return displ_board

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for name, piece in self.components.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1
        for name, mask in self.masks.items():
            x, y = self.masks['boundary'].get_positions()
            z = np.repeat(layer, len(x))
            displ_board[(z, x, y)] = 1
            layer += 1
        return displ_board


def addTuple(a, b):
    return tuple(sum(x) for x in zip(a, b))
