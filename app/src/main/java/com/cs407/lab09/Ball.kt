//package com.cs407.lab09
//
///**
// * Represents a ball that can move. (No Android UI imports!)
// *
// * Constructor parameters:
// * - backgroundWidth: the width of the background, of type Float
// * - backgroundHeight: the height of the background, of type Float
// * - ballSize: the width/height of the ball, of type Float
// */
//class Ball(
//    private val backgroundWidth: Float,
//    private val backgroundHeight: Float,
//    private val ballSize: Float
//) {
//    var posX = 0f
//    var posY = 0f
//    var velocityX = 0f
//    var velocityY = 0f
//    private var accX = 0f
//    private var accY = 0f
//
//    private var isFirstUpdate = true
//
//    init {
//        // TODO: Call reset()
//    }
//
//    /**
//     * Updates the ball's position and velocity based on the given acceleration and time step.
//     * (See lab handout for physics equations)
//     */
//    fun updatePositionAndVelocity(xAcc: Float, yAcc: Float, dT: Float) {
//        if(isFirstUpdate) {
//            isFirstUpdate = false
//            accX = xAcc
//            accY = yAcc
//            return
//        }
//
//    }
//
//    /**
//     * Ensures the ball does not move outside the boundaries.
//     * When it collides, velocity and acceleration perpendicular to the
//     * boundary should be set to 0.
//     */
//    fun checkBoundaries() {
//        // TODO: implement the checkBoundaries function
//        // (Check all 4 walls: left, right, top, bottom)
//    }
//
//    /**
//     * Resets the ball to the center of the screen with zero
//     * velocity and acceleration.
//     */
//    fun reset() {
//        // TODO: implement the reset function
//        // (Reset posX, posY, velocityX, velocityY, accX, accY, isFirstUpdate)
//    }
//}


package com.cs407.lab09

/**
 * Represents a ball that can move. (No Android UI imports!)
 *
 * Constructor parameters:
 * - backgroundWidth: the width of the background, of type Float
 * - backgroundHeight: the height of the background, of type Float
 * - ballSize: the width/height of the ball, of type Float
 */
class Ball(
    private val backgroundWidth: Float,
    private val backgroundHeight: Float,
    private val ballSize: Float
) {
    // Top-left position of the ball on the field (screen coordinates)
    var posX = 0f
    var posY = 0f

    // Velocity in x/y directions
    var velocityX = 0f
    var velocityY = 0f

    // Previous acceleration values (used for linear interpolation)
    private var accX = 0f
    private var accY = 0f

    private var isFirstUpdate = true

    init {
        // Start in the center of the field
        reset()
    }

    /**
     * Updates the ball's position and velocity based on the given acceleration and time step.
     * (See lab handout for physics equations)
     *
     * a0 = previous acceleration, a1 = current acceleration
     * v1 = v0 + 0.5 * (a0 + a1) * dt
     * l = v0 * dt + (1/6) * dt^2 * (3 * a0 + a1)
     */
    fun updatePositionAndVelocity(xAcc: Float, yAcc: Float, dT: Float) {
        // On the very first update, just store the acceleration and don't move yet
        if (isFirstUpdate) {
            isFirstUpdate = false
            accX = xAcc
            accY = yAcc
            return
        }

        val dt = dT

        // --- X direction ---
        val a0x = accX
        val a1x = xAcc

        val newVx = velocityX + 0.5f * (a0x + a1x) * dt
        val dx = velocityX * dt + (1f / 6f) * dt * dt * (3f * a0x + a1x)

        // --- Y direction ---
        val a0y = accY
        val a1y = yAcc

        val newVy = velocityY + 0.5f * (a0y + a1y) * dt
        val dy = velocityY * dt + (1f / 6f) * dt * dt * (3f * a0y + a1y)

        // Update position
        posX += dx
        posY += dy

        // Update velocity and last acceleration
        velocityX = newVx
        velocityY = newVy
        accX = a1x
        accY = a1y

        // Enforce boundaries after moving
        checkBoundaries()
    }

    /**
     * Ensures the ball does not move outside the boundaries.
     * When it collides, velocity and acceleration perpendicular to the
     * boundary should be set to 0.
     */
    fun checkBoundaries() {
        val maxX = backgroundWidth - ballSize
        val maxY = backgroundHeight - ballSize

        // Left wall
        if (posX < 0f) {
            posX = 0f
            velocityX = 0f
            accX = 0f
        }
        // Right wall
        if (posX > maxX) {
            posX = maxX
            velocityX = 0f
            accX = 0f
        }
        // Top wall
        if (posY < 0f) {
            posY = 0f
            velocityY = 0f
            accY = 0f
        }
        // Bottom wall
        if (posY > maxY) {
            posY = maxY
            velocityY = 0f
            accY = 0f
        }
    }

    /**
     * Resets the ball to the center of the screen with zero
     * velocity and acceleration.
     */
    fun reset() {
        posX = (backgroundWidth - ballSize) / 2f
        posY = (backgroundHeight - ballSize) / 2f

        velocityX = 0f
        velocityY = 0f
        accX = 0f
        accY = 0f

        isFirstUpdate = true
    }
}
