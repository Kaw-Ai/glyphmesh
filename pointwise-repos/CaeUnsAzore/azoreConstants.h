/****************************************************************************
 *
 * (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
 *
 * This sample source code is not supported by Cadence Design Systems, Inc.
 * It is provided freely for demonstration purposes only.
 * SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
 *
 ***************************************************************************/
/****************************************************************************
 *
 * AZORE constants
 *
 ***************************************************************************/

#ifndef _AZORECONSTANTS_H_
#define _AZORECONSTANTS_H_

enum SectionId {
    AZORE_CELLS = 12,
    AZORE_FACES = 13,
    AZORE_NODES = 10,
    AZORE_ZONE = 45,             // region where a BC is defined
    AZORE_COMMENT = 0,
    AZORE_HEADER = 1,
    AZORE_DIMENSION = 2,
    AZORE_PERIODIC_SHADOW = 18,
    AZORE_INTERIOR = 2,          // interior face
};

enum CellType {
    AZORE_CELL_TRI = 1,
    AZORE_CELL_QUAD = 3,
    AZORE_CELL_TET = 2,
    AZORE_CELL_WEDGE = 6,
    AZORE_CELL_HEX = 4,
    AZORE_CELL_PYR = 5,
    AZORE_CELL_MIXED = 0,
    AZORE_CELL_OTHER = 7,
};

enum FaceType {
    AZORE_FACE_BAR = 2,
    AZORE_FACE_TRI = 3,
    AZORE_FACE_QUAD = 4,
    AZORE_FACE_MIXED = 0,
    AZORE_FACE_OTHER = 5,
};

#endif

/****************************************************************************
 *
 * This file is licensed under the Cadence Public License Version 1.0 (the
 * "License"), a copy of which is found in the included file named "LICENSE",
 * and is distributed "AS IS." TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE
 * LAW, CADENCE DISCLAIMS ALL WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO
 * ANY PARTY FOR ANY DAMAGES ARISING OUT OF OR RELATING TO USE OF THIS FILE.
 * Please see the License for the full text of applicable terms.
 *
 ****************************************************************************/
