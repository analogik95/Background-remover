package com.vidsticker.app.matting

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.random.Random

private const val W = 60
private const val H = 50
private val SCREEN = Triple(0, 177, 64) // matches the desktop test fixture's green screen
private val SUBJECT = Triple(222, 96, 40)

/** Green screen with an orange square at a given x offset - mirrors the desktop suite's `make_frame`. */
private fun makeFrame(xOffset: Int = 0): RgbImage {
    val r = FloatArray(W * H); val g = FloatArray(W * H); val b = FloatArray(W * H)
    for (y in 0 until H) for (x in 0 until W) {
        val i = y * W + x
        val inSquare = x in (10 + xOffset until 10 + xOffset + 20) && y in 15 until 35
        val (cr, cg, cb) = if (inSquare) SUBJECT else SCREEN
        r[i] = cr.toFloat(); g[i] = cg.toFloat(); b[i] = cb.toFloat()
    }
    return RgbImage(W, H, r, g, b)
}

/** The mask a perfect AI matcher would produce for [makeFrame]'s square. */
private fun perfectMask(xOffset: Int = 0): FloatArray {
    val m = FloatArray(W * H)
    for (y in 0 until H) for (x in 0 until W) {
        if (x in (10 + xOffset until 10 + xOffset + 20) && y in 15 until 35) m[y * W + x] = 1f
    }
    return m
}

class MattingTest {

    // ---- colour space, verified against cv2.cvtColor(_, COLOR_RGB2YCrCb) -----

    @Test fun `rgbToYCrCb matches OpenCV on known samples`() {
        // (r, g, b) -> (Y, Cr, Cb), captured from cv2 during the desktop port.
        val cases = listOf(
            Triple(200, 50, 80) to Triple(98, 201, 118),
            Triple(10, 200, 60) to Triple(127, 45, 90),
            Triple(240, 240, 240) to Triple(240, 128, 128),
            Triple(5, 5, 5) to Triple(5, 128, 128),
            Triple(0, 177, 64) to Triple(111, 49, 101),
        )
        for ((rgb, expected) in cases) {
            val (y, cr, cb) = Matting.rgbToYCrCb(rgb.first.toFloat(), rgb.second.toFloat(), rgb.third.toFloat())
            assertEquals("Y for $rgb", expected.first.toFloat(), y, 0.01f)
            assertEquals("Cr for $rgb", expected.second.toFloat(), cr, 0.51f)
            assertEquals("Cb for $rgb", expected.third.toFloat(), cb, 0.51f)
        }
    }

    // ---- key detection --------------------------------------------------------

    @Test fun `detectKey finds the green screen`() {
        val key = Matting.detectKey(makeFrame())
        assertNotNull(key)
        val (_, cr, cb) = Matting.rgbToYCrCb(SCREEN.first.toFloat(), SCREEN.second.toFloat(), SCREEN.third.toFloat())
        assertEquals(cr, key!!.cr, 2f)
        assertEquals(cb, key.cb, 2f)
    }

    @Test fun `detectKey declines on a plain grey backdrop`() {
        val flat = RgbImage(20, 20, FloatArray(400) { 130f }, FloatArray(400) { 130f }, FloatArray(400) { 130f })
        assertNull(Matting.detectKey(flat))
    }

    @Test fun `detectKey declines on a busy backdrop`() {
        val rnd = Random(0)
        val n = 80 * 80
        val img = RgbImage(80, 80,
            FloatArray(n) { rnd.nextInt(256).toFloat() },
            FloatArray(n) { rnd.nextInt(256).toFloat() },
            FloatArray(n) { rnd.nextInt(256).toFloat() })
        assertNull(Matting.detectKey(img))
    }

    // ---- chroma alpha -----------------------------------------------------

    @Test fun `chromaAlpha separates subject from screen`() {
        val img = makeFrame()
        val key = Matting.detectKey(img)!!
        val alpha = Matting.chromaAlpha(img, key, tolerance = 8f, softness = 20f)
        assertEquals(0f, alpha[5 * W + 5], 0.001f) // screen corner
        assertEquals(1f, alpha[20 * W + 15], 0.001f) // inside the square
    }

    @Test fun `chromaAlpha ramps smoothly between tolerance and softness`() {
        val img = makeFrame()
        val key = Matting.detectKey(img)!!
        val alpha = Matting.chromaAlpha(img, key, tolerance = 8f, softness = 20f)
        // Screen pixels (distance ~0) are 0; subject pixels (large distance) are 1;
        // nothing in between exists in this fixture, so just check the two extremes.
        assertEquals(0f, alpha[0], 0.01f)
        assertTrue(alpha[20 * W + 15] > 0.9f)
    }

    // ---- ramp calibration ---------------------------------------------------

    // calibrateRamp erodes both masks by a 15x15 structuring element and
    // requires >=500 surviving pixels on each side - the desktop tool's real
    // video frames clear that easily, but the shared 60x50 fixture used
    // elsewhere in this file cannot, so this test gets its own, larger one.
    private val bigW = 240
    private val bigH = 200

    private fun makeBigFrame(): RgbImage {
        val r = FloatArray(bigW * bigH); val g = FloatArray(bigW * bigH); val b = FloatArray(bigW * bigH)
        for (y in 0 until bigH) for (x in 0 until bigW) {
            val i = y * bigW + x
            val inSquare = x in 60 until 180 && y in 40 until 160
            val (cr, cg, cb) = if (inSquare) SUBJECT else SCREEN
            r[i] = cr.toFloat(); g[i] = cg.toFloat(); b[i] = cb.toFloat()
        }
        return RgbImage(bigW, bigH, r, g, b)
    }

    private fun bigMask(): FloatArray {
        val m = FloatArray(bigW * bigH)
        for (y in 0 until bigH) for (x in 0 until bigW) if (x in 60 until 180 && y in 40 until 160) m[y * bigW + x] = 1f
        return m
    }

    @Test fun `calibrateRamp reaches full opacity at or before the subject`() {
        val img = makeBigFrame()
        val key = Matting.detectKey(img)!!
        val fitted = Matting.calibrateRamp(img, key, bigMask())
        assertNotNull(fitted)
        val (tolerance, softness) = fitted!!
        assertTrue("tolerance should be positive", tolerance >= 6f)
        assertTrue("softness should be positive", softness >= 15f)

        // The ramp must not saturate before the subject's own chroma distance -
        // that is the desktop bug this exists to prevent (a halo on motion blur).
        val mask = bigMask()
        val subjectDistances = (0 until bigW * bigH).filter { mask[it] > 0.5f }.map {
            val (_, cr, cb) = Matting.rgbToYCrCb(img.r[it], img.g[it], img.b[it])
            hypot((cr - key.cr).toDouble(), (cb - key.cb).toDouble())
        }
        val minSubjectDistance = subjectDistances.min()
        assertTrue(
            "ramp saturates at ${tolerance + softness}, before subject distance $minSubjectDistance",
            tolerance + softness <= minSubjectDistance + 1.0,
        )
    }

    @Test fun `calibrateRamp declines without enough of either side`() {
        val img = makeBigFrame()
        val key = Matting.detectKey(img)!!
        assertNull(Matting.calibrateRamp(img, key, FloatArray(bigW * bigH) { 1f }))
        assertNull(Matting.calibrateRamp(img, key, FloatArray(bigW * bigH) { 0f }))
    }

    // ---- combine / compute ----------------------------------------------------

    @Test fun `combineMin takes the stricter of two alphas`() {
        val a = floatArrayOf(0f, 1f, 0.5f)
        val b = floatArrayOf(1f, 0f, 0.7f)
        assertEquals(listOf(0f, 0f, 0.5f), Matting.combineMin(a, b).toList())
    }

    @Test fun `computeAlpha hybrid never exceeds the chroma key`() {
        val img = makeFrame()
        val key = Matting.detectKey(img)!!
        // A neural matte that claims everything is subject must not override the key.
        val allOnes = FloatArray(W * H) { 1f }
        val alpha = Matting.computeAlpha(img, MatteConfig(mode = MatteMode.HYBRID), allOnes, key)
        assertEquals(0f, alpha[0], 0.01f)
        assertTrue(alpha[20 * W + 15] > 0.9f)
    }

    @Test(expected = IllegalStateException::class)
    fun `computeAlpha chroma mode without a key throws`() {
        val img = makeFrame()
        Matting.computeAlpha(img, MatteConfig(mode = MatteMode.CHROMA), null, null)
    }

    // ---- screen unmixing --------------------------------------------------

    @Test fun `unmixScreen recovers the subject colour from a half-covered pixel`() {
        val screen = floatArrayOf(SCREEN.first.toFloat(), SCREEN.second.toFloat(), SCREEN.third.toFloat())
        val subject = floatArrayOf(SUBJECT.first.toFloat(), SUBJECT.second.toFloat(), SUBJECT.third.toFloat())
        val observed = RgbImage(1, 1,
            floatArrayOf((subject[0] + screen[0]) / 2f),
            floatArrayOf((subject[1] + screen[1]) / 2f),
            floatArrayOf((subject[2] + screen[2]) / 2f))
        val alpha = floatArrayOf(0.5f)

        val out = Matting.unmixScreen(observed, alpha, screen, amount = 1f)
        assertEquals(subject[0], out.r[0], 2f)
        assertEquals(subject[1], out.g[0], 2f)
        assertEquals(subject[2], out.b[0], 2f)
    }

    @Test fun `unmixScreen amount zero is a no-op`() {
        val img = makeFrame()
        val out = Matting.unmixScreen(img, FloatArray(W * H) { 0.5f }, floatArrayOf(0f, 177f, 64f), amount = 0f)
        assertTrue(out === img)
    }

    @Test fun `screenRgb reads the backdrop colour`() {
        val img = makeFrame()
        // perfectMask is 1 (opaque) inside the square and 0 (transparent) on the
        // screen - exactly what screenRgb expects to find the backdrop from.
        val screen = Matting.screenRgb(img, perfectMask())
        assertNotNull(screen)
        assertEquals(SCREEN.first.toFloat(), screen!![0], 1f)
        assertEquals(SCREEN.second.toFloat(), screen[1], 1f)
        assertEquals(SCREEN.third.toFloat(), screen[2], 1f)
    }

    @Test fun `screenRgb is null without enough background`() {
        assertNull(Matting.screenRgb(makeFrame(), FloatArray(W * H) { 1f }))
    }

    // ---- temporal despeckling -----------------------------------------------

    @Test fun `temporalDespeckle kills a one-frame spike`() {
        val zero = FloatArray(4) { 0f }
        val one = FloatArray(4) { 1f }
        assertEquals(0f, Matting.temporalDespeckle(zero, one, zero).max())
    }

    @Test fun `temporalDespeckle never adds coverage beyond the current frame`() {
        // The bug this replaced: neighbours that agree can out-vote the current
        // frame under a plain median, painting a moving limb onto backdrop.
        val zero = FloatArray(4) { 0f }
        val one = FloatArray(4) { 1f }
        assertEquals(0f, Matting.temporalDespeckle(one, zero, one).max())

        val rnd = Random(0)
        val p = FloatArray(50) { rnd.nextFloat() }
        val c = FloatArray(50) { rnd.nextFloat() }
        val n = FloatArray(50) { rnd.nextFloat() }
        val out = Matting.temporalDespeckle(p, c, n)
        for (i in out.indices) assertTrue(out[i] <= c[i] + 1e-6f)
    }

    @Test fun `temporalDespeckle passes through unchanged with a missing neighbour`() {
        val cur = floatArrayOf(0.3f, 0.7f)
        assertEquals(cur.toList(), Matting.temporalDespeckle(null, cur, floatArrayOf(0f, 0f)).toList())
    }

    // ---- bounding box ----------------------------------------------------------

    @Test fun `contentBoundingBox is tight`() {
        val a = FloatArray(50 * 60)
        for (y in 10 until 20) for (x in 30 until 45) a[y * 50 + x] = 1f
        val box = Matting.contentBoundingBox(a, 50, 60)
        assertEquals(listOf(30, 10, 45, 20), box!!.toList())
    }

    @Test fun `contentBoundingBox of an empty matte is null`() {
        assertNull(Matting.contentBoundingBox(FloatArray(100), 10, 10))
    }

    // ---- premultiplied resize -------------------------------------------------

    @Test fun `resizeRgbaPremultiplied does not drag the backdrop into the edge`() {
        val n = 32 * 32
        val r = FloatArray(n) { SCREEN.first.toFloat() }
        val g = FloatArray(n) { SCREEN.second.toFloat() }
        val b = FloatArray(n) { SCREEN.third.toFloat() }
        val a = FloatArray(n)
        for (y in 5 until 27) for (x in 5 until 27) {
            val i = y * 32 + x
            r[i] = SUBJECT.first.toFloat(); g[i] = SUBJECT.second.toFloat(); b[i] = SUBJECT.third.toFloat()
            a[i] = 255f
        }
        val out = Matting.resizeRgbaPremultiplied(r, g, b, a, 32, 32, 11, 11)
        var maxShift = 0f
        for (i in out.a.indices) {
            if (out.a[i] > 200f) {
                maxShift = maxOf(
                    maxShift,
                    abs(out.r[i] - SUBJECT.first), abs(out.g[i] - SUBJECT.second), abs(out.b[i] - SUBJECT.third),
                )
            }
        }
        assertEquals("no screen colour should bleed into a fully-opaque output pixel", 0f, maxShift, 0.01f)
    }

    @Test fun `resizeRgbaPremultiplied preserves requested output size`() {
        val n = 10 * 20
        val out = Matting.resizeRgbaPremultiplied(
            FloatArray(n), FloatArray(n), FloatArray(n), FloatArray(n) { 255f }, 20, 10, 10, 5,
        )
        assertEquals(50, out.a.size)
    }
}
