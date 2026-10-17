"""Offscreen OpenGL renderer (moderngl): sky with a ray-traced planet, low-poly scene, bloom.

Runs headless through EGL, on Mesa's software rasterizer when there's no GPU (as on a
plain VPS), at roughly 0.1 s per 1080x1920 frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl
import numpy as np

from whatif.meshes import PROTOTYPES

VERTEX = """
#version 330
uniform mat4 u_viewproj;
in vec3 in_pos;
in mat4 in_model;
in vec4 in_color;
in vec4 in_params;
out vec3 v_world;
out vec3 v_local;
flat out vec4 v_color;
flat out vec4 v_params;
void main() {
    vec4 w = in_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_local = in_pos;
    v_color = in_color;
    v_params = in_params;
    gl_Position = u_viewproj * w;
}
"""

FRAGMENT = """
#version 330
uniform vec3 u_cam;
uniform float u_time;
uniform vec3 u_moon_dir;
uniform vec3 u_moon_col;
uniform vec3 u_body_dir;
uniform vec3 u_body_col;
uniform vec3 u_amb_sky;
uniform vec3 u_amb_ground;
uniform vec3 u_fog_col;
uniform float u_fog_density;
uniform float u_power;
uniform float u_lava;
uniform vec4 u_lpos[8];   // xyz, radius
uniform vec4 u_lcol[8];
uniform int u_nlights;
uniform sampler2DShadow u_shadow_map;
uniform mat4 u_shadow_vp;
uniform int u_use_shadow;
uniform float u_snow;
uniform float u_dust_on;
uniform float u_front_z;
uniform float u_dust_h;
uniform vec3 u_dust_col;
uniform float u_frost;
uniform float u_ice;
uniform float u_dry;      // heat: greens fade to straw
uniform float u_scorch;   // heat: everything chars, in patches
uniform float u_boil;     // heat: water churns and pales
in vec3 v_world;
in vec3 v_local;
flat in vec4 v_color;
flat in vec4 v_params;
out vec4 f_color;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash13(vec3(i, 1.0)), b = hash13(vec3(i + vec2(1, 0), 1.0));
    float c = hash13(vec3(i + vec2(0, 1), 1.0)), d = hash13(vec3(i + vec2(1, 1), 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// The air blast's dust wall: a billowing front at z = u_front_z, everything beyond it buried.
// Returns how much dust covers the point where the ray from the camera through `p` crosses the front.
float dust_cover(vec3 cam, vec3 dir) {
    if (u_dust_on < 0.5 || dir.z > -1e-4) return 0.0;
    float s = (u_front_z - cam.z) / dir.z;
    if (s <= 0.0) return 1.0;  // the front has reached us
    vec3 q = cam + dir * s;
    float billow = 0.65 * vnoise(vec2(q.x / 260.0, u_time * 0.6)) + 0.35 * vnoise(vec2(q.x / 90.0 + 7.0, u_time * 1.3));
    float top = u_dust_h * (0.55 + 0.6 * billow);
    return 1.0 - smoothstep(top - 40.0, top + 10.0, q.y);
}

vec3 dust_color(vec3 cam, vec3 dir) {
    float s = (u_front_z - cam.z) / min(dir.z, -1e-4);
    vec3 q = cam + dir * max(s, 0.0);
    float n = vnoise(vec2(q.x / 120.0, q.y / 80.0 - u_time * 0.8));
    float height = clamp(q.y / max(u_dust_h, 1.0), 0.0, 1.0);
    return u_dust_col * (0.6 + 0.35 * n + 0.35 * height);
}

float shadow_at(vec3 n) {
    if (u_use_shadow == 0) return 1.0;
    vec4 s = u_shadow_vp * vec4(v_world + n * 0.5, 1.0);
    vec3 p = s.xyz / s.w * 0.5 + 0.5;
    if (p.x <= 0.0 || p.x >= 1.0 || p.y <= 0.0 || p.y >= 1.0 || p.z >= 1.0) return 1.0;
    float sum = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            sum += texture(u_shadow_map, vec3(p.xy + vec2(i, j) * texel, p.z - 0.0008));
    return sum / 25.0;
}

vec3 point_lights(vec3 n) {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < u_nlights; i++) {
        vec3 L = u_lpos[i].xyz - v_world;
        float d = length(L);
        float att = max(1.0 - d / u_lpos[i].w, 0.0);
        sum += u_lcol[i].rgb * att * att * (0.25 + 0.75 * max(dot(n, L / max(d, 1e-3)), 0.0));
    }
    return sum;
}

void main() {
    vec3 n = normalize(cross(dFdx(v_world), dFdy(v_world)));
    vec3 V = normalize(u_cam - v_world);
    if (dot(n, V) < 0.0) n = -n;
    int mat = int(v_params.x + 0.5);
    vec3 albedo = v_color.rgb;
    float emissive = v_color.a;
    float dist = length(u_cam - v_world);
    vec3 col;

    if (mat == 1) {  // water
        vec2 p = v_world.xz;
        float t = u_time;
        float fade = clamp(1.0 - dist / 900.0, 0.0, 1.0);
        vec2 g = vec2(0.0);
        // A few wave trains in different directions (no grid pattern), fading out with distance.
        const vec4 waves[6] = vec4[](vec4(0.95, 0.31, 0.21, 0.07), vec4(0.17, 0.98, 0.33, 0.05),
                                     vec4(-0.6, 0.8, 0.52, 0.04), vec4(-0.97, -0.24, 0.81, 0.03),
                                     vec4(0.55, -0.83, 1.27, 0.022), vec4(0.31, 0.95, 1.93, 0.016));
        for (int i = 0; i < 6; i++) {
            float ph = dot(waves[i].xy, p) * waves[i].z + t * sqrt(9.8 * waves[i].z);
            g += waves[i].xy * waves[i].w * cos(ph) * (i < 3 ? 1.0 : fade);
        }
        vec3 wn = normalize(vec3(g.x * (0.4 + 0.6 * fade), 1.0, g.y * (0.4 + 0.6 * fade)));
        float fres = 0.03 + 0.97 * pow(1.0 - max(dot(wn, V), 0.0), 5.0);
        vec3 R = reflect(-V, wn);
        vec3 refl = mix(u_fog_col, u_amb_sky * 1.6, clamp(R.y * 3.0, 0.0, 1.0));
        float sm = pow(max(dot(R, u_moon_dir), 0.0), 900.0) * 10.0 + pow(max(dot(R, u_moon_dir), 0.0), 60.0) * 0.22;
        float sb = pow(max(dot(R, u_body_dir), 0.0), 160.0) * 1.2 + pow(max(dot(R, u_body_dir), 0.0), 14.0) * 0.1;
        col = albedo * (u_amb_sky * 0.5 + u_body_col * 0.12) + refl * fres + u_moon_col * sm + u_body_col * sb;
        col += albedo * point_lights(wn) * 2.0;
        if (u_boil > 0.0) {   // boiling: churning white foam over the whole surface
            float foam = smoothstep(0.55, 0.9, vnoise(p * 0.9 + vec2(t * 1.7, -t * 1.3)) * 0.6
                                    + hash13(floor(vec3(p * 2.2, floor(t * 12.0)))) * 0.5);
            col = mix(col, vec3(0.85, 0.86, 0.88) * (u_amb_sky + u_moon_col * 0.6), foam * u_boil);
        }
        if (u_ice > 0.0) {   // frozen over: pale, rough, barely reflective
            vec3 ice = vec3(0.62, 0.7, 0.78) * (u_amb_sky * 0.9 + u_moon_col * max(u_moon_dir.y, 0.0) + 0.02);
            col = mix(col, ice + refl * 0.08, u_ice);
        }
    } else if (mat == 2) {  // pure light: bulbs, windows, meteors
        float powered = mix(1.0, u_power, v_params.w);
        col = albedo * emissive * powered;
    } else if (mat == 6) {  // flame, hot at the base
        float y = clamp(v_local.y, 0.0, 1.0);
        vec3 hot = mix(vec3(1.0, 0.85, 0.45), vec3(1.0, 0.35, 0.06), y);
        col = hot * emissive * (1.2 - 0.6 * y);
    } else {
        if (mat == 4) {  // stripes: along local x, around the y axis, or wooden planks along z
            if (v_params.w > 1.5) {
                float s = (v_local.z + 0.5) * v_params.z;
                albedo *= 1.0 - 0.28 * step(0.82, fract(s)) * clamp(1.0 - dist / 70.0, 0.0, 1.0);
            } else {
                float s = v_params.w > 0.5 ? atan(v_local.z, v_local.x) / 6.2831853 * v_params.z
                                           : (v_local.x + 0.5) * v_params.z;
                albedo = mix(albedo, vec3(0.92, 0.9, 0.86), step(0.5, fract(s)));
            }
        }
        if (mat == 8) {  // terrain: meadow, forest floor, rock and snow by height and slope
            vec2 q = v_world.xz;
            float h = v_world.y;
            float steep = 1.0 - abs(n.y);
            float k = 0.5 + 0.5 * sin(q.x * 0.013 + sin(q.y * 0.011) * 2.0) * sin(q.y * 0.017 + 1.0);
            vec3 forest = mix(vec3(0.045, 0.075, 0.035), vec3(0.07, 0.1, 0.045), k);
            vec3 meadow = mix(vec3(0.15, 0.19, 0.06), vec3(0.22, 0.23, 0.08), k);
            albedo = mix(meadow, forest, smoothstep(16.0, 34.0, h));
            albedo = mix(albedo, vec3(0.16, 0.145, 0.13), smoothstep(0.3, 0.48, steep));
            float line = u_snow + 70.0 * sin(q.x * 0.0035 + q.y * 0.001);
            albedo = mix(albedo, vec3(0.86, 0.89, 0.93), smoothstep(line, line + 70.0, h) * (1.0 - smoothstep(0.5, 0.72, steep)));
            albedo = mix(albedo, vec3(0.22, 0.18, 0.12), 1.0 - smoothstep(0.4, 2.5, h));
        }
        if (mat == 7) {  // ground: sand, rock and grass with soft patches
            vec2 q = v_world.xz;
            float k = 0.5 * sin(q.x * 0.11 + sin(q.y * 0.07) * 2.0) + 0.5 * sin(q.y * 0.13 + sin(q.x * 0.05) * 3.0);
            float fine = hash13(floor(vec3(q * 1.5, 0.0)));
            albedo *= 0.86 + 0.1 * k + 0.08 * fine * clamp(1.0 - dist / 60.0, 0.0, 1.0);
        }
        albedo = mix(albedo, vec3(0.8, 0.85, 0.9), u_frost * smoothstep(0.15, 0.7, n.y));   // frost settles on tops
        if (u_dry > 0.0) {   // plants dry out: anything green turns to straw
            float green = clamp((albedo.g - max(albedo.r, albedo.b)) * 6.0, 0.0, 1.0);
            vec3 straw = vec3(0.36, 0.28, 0.12) * (0.6 + 0.6 * dot(albedo, vec3(0.33)) / max(albedo.g, 0.02));
            albedo = mix(albedo, straw, green * u_dry);
        }
        if (u_scorch > 0.0 && !(mat == 7 && v_params.w > 0.5)) {  // charring spreads in patches until everything is black (p2=1 ground, e.g. sand, never chars)
            float patch = vnoise(v_world.xz * 0.35 + v_world.y * 0.2) * 0.6 + hash13(floor(v_world * 1.7)) * 0.4;
            float burnt = smoothstep(patch - 0.15, patch + 0.15, u_scorch * 1.3 - 0.15);
            albedo = mix(albedo, vec3(0.025, 0.02, 0.018), burnt);
        }
        float hemi = n.y * 0.5 + 0.5;
        vec3 amb = mix(u_amb_ground, u_amb_sky, hemi);
        vec3 diff = u_moon_col * max(dot(n, u_moon_dir), 0.0) * shadow_at(n) + u_body_col * max(dot(n, u_body_dir), 0.0);
        col = albedo * (amb + diff + point_lights(n)) + albedo * emissive;
        if (mat == 3 && abs(n.y) < 0.5) {  // building windows
            bool side_x = abs(n.x) > abs(n.z);
            float u = side_x ? v_world.z : v_world.x;
            vec2 cell = vec2(u / 2.7, (v_world.y - 2.2) / 3.2);
            vec2 f = fract(cell);
            vec2 id = floor(cell);
            float inside = step(0.3, f.x) * step(f.x, 0.7) * step(0.3, f.y) * step(f.y, 0.74) * step(0.0, id.y);
            float h = hash13(vec3(id, v_params.y * 7.31 + (side_x ? 3.7 : 0.0)));
            float lit = step(h, v_params.z * u_power);
            vec3 warm = mix(vec3(1.0, 0.62, 0.3), vec3(1.0, 0.85, 0.6), hash13(vec3(id.yx, v_params.y)));
            vec3 window = lit > 0.5 ? warm * 1.5 : albedo * 0.1 + u_amb_sky * 0.15;
            col = mix(col, window, inside);
        }
        if (mat == 5) {  // mountains: lava runs down from the peaks
            float rivers = smoothstep(0.86, 0.97, sin(atan(v_local.z, v_local.x) * 7.0 + v_params.y + v_local.y * 4.0));
            float glow = u_lava * smoothstep(0.3, 0.95, v_local.y) * (0.2 + rivers) + u_lava * smoothstep(0.88, 0.98, v_local.y);
            col += vec3(1.0, 0.32, 0.05) * glow * 4.0;
        }
    }

    float fog = 1.0 - exp(-dist * u_fog_density);
    if (mat == 2) fog *= 0.6;
    col = mix(col, u_fog_col, clamp(fog, 0.0, 1.0));
    if (u_dust_on > 0.5 && v_world.z < u_front_z) {
        vec3 d = normalize(v_world - u_cam);
        col = mix(col, dust_color(u_cam, d), dust_cover(u_cam, d));
    }
    f_color = vec4(col, 1.0);
}
"""

FULLSCREEN = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
out vec2 v_ndc;
void main() {
    v_ndc = in_pos;
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SKY = """
#version 330
uniform mat4 u_inv_viewproj;
uniform vec3 u_cam;
uniform vec3 u_zenith;
uniform vec3 u_mid;
uniform vec3 u_horizon;
uniform float u_stars;
uniform float u_px;
uniform vec3 u_sun_dir;
uniform vec3 u_moon_dir;
uniform float u_moon_r;
uniform vec3 u_body_dir;
uniform float u_body_d;      // distance to the body's center, in body radii
uniform mat3 u_body_frame;   // columns: ring-plane x, pole, ring-plane z (world space)
uniform float u_flat;        // polar / equatorial radius
uniform vec2 u_rings;        // inner, outer radius (0 = no rings)
uniform vec3 u_band_a;
uniform vec3 u_band_b;
uniform vec3 u_band_c;
uniform vec3 u_ring_col;
uniform float u_body_bright;
uniform vec3 u_glow_col;
uniform float u_glow;
uniform float u_time;
uniform vec3 u_cloud;
uniform int u_body_type;     // 0 planet, 1 black hole, 2 comet, 3 star, 4 pulsar
uniform vec2 u_disk;         // black hole: accretion disk inner and outer radius, in horizon radii
uniform float u_disk_bright;
uniform float u_bh_veil;
uniform float u_moon_vis;
uniform int u_ring_style;
uniform vec3 u_star_col;
uniform vec3 u_spot;        // a great storm: latitude (sin), longitude (radians), size (0 = none)
uniform vec3 u_spot_col;
uniform float u_surface;     // 0 = banded giant, 1 = rocky (mottled terrain, small polar caps)
uniform float u_sun_show;   // escapes: our Sun, shrinking to a star
uniform float u_sun_r;
uniform float u_sun_b;
uniform float u_sun_glow;
uniform float u_milky;      // the Milky Way's band across the night sky
uniform float u_galaxy;     // the whole galaxy, seen from outside
uniform float u_galaxy_r;
uniform vec3 u_galaxy_dir;
uniform float u_cloud_amt;
uniform float u_star_bright;
// Comet (body type 2) and what it does when it lands.
uniform float u_coma_r;
uniform float u_comet_bright;
uniform vec3 u_tail_dir;
uniform float u_tail_len;
uniform vec3 u_tail_col;
uniform float u_entry;
uniform vec3 u_trail_dir;
uniform float u_trail_len;
uniform vec3 u_site_dir;
uniform float u_fireball;
uniform float u_fireball_r;
uniform float u_plume;
uniform vec3 u_plume_lit;
uniform float u_dust_on;
uniform float u_front_z;
uniform float u_dust_h;
uniform vec3 u_dust_col;
uniform vec3 u_cam_world;
uniform vec3 u_beam_dir;     // pulsar: one beam's direction in space (the other points the opposite way)
uniform vec3 u_beam_col;
uniform float u_beam_len;    // how far the beams reach across the sky, in radians
uniform float u_beam_bright;
uniform float u_pulse;       // 1 while a beam points straight at us
in vec2 v_ndc;
out vec4 f_color;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float hnoise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * hnoise2(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        a *= 0.5;
    }
    return v;
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash13(vec3(i, 1.0)), b = hash13(vec3(i + vec2(1, 0), 1.0));
    float c = hash13(vec3(i + vec2(0, 1), 1.0)), d = hash13(vec3(i + vec2(1, 1), 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// The air blast's dust wall: a billowing front at z = u_front_z, everything beyond it buried.
// Returns how much dust covers the point where the ray from the camera through `p` crosses the front.
float dust_cover(vec3 cam, vec3 dir) {
    if (u_dust_on < 0.5 || dir.z > -1e-4) return 0.0;
    float s = (u_front_z - cam.z) / dir.z;
    if (s <= 0.0) return 1.0;  // the front has reached us
    vec3 q = cam + dir * s;
    float billow = 0.65 * vnoise(vec2(q.x / 260.0, u_time * 0.6)) + 0.35 * vnoise(vec2(q.x / 90.0 + 7.0, u_time * 1.3));
    float top = u_dust_h * (0.55 + 0.6 * billow);
    return 1.0 - smoothstep(top - 40.0, top + 10.0, q.y);
}

vec3 dust_color(vec3 cam, vec3 dir) {
    float s = (u_front_z - cam.z) / min(dir.z, -1e-4);
    vec3 q = cam + dir * max(s, 0.0);
    float n = vnoise(vec2(q.x / 120.0, q.y / 80.0 - u_time * 0.8));
    float height = clamp(q.y / max(u_dust_h, 1.0), 0.0, 1.0);
    return u_dust_col * (0.6 + 0.35 * n + 0.35 * height);
}

// Optical depth of the rings at radius r (in body radii), with soft edges of width w.
float narrow(float r, float c, float width, float fw) {
    float s = max(width, fw);   // wider than a pixel: fade it instead of letting it shimmer
    return exp(-pow((r - c) / s, 2.0)) * min(width / s, 1.0);
}

float ring_alpha(float r, float w) {
    if (u_ring_style == 1) {   // Uranus: nine thin dark rings and the wider epsilon ring
        float u = 0.5 * narrow(r, 1.637, 0.002, w) + 0.5 * narrow(r, 1.652, 0.002, w) + 0.5 * narrow(r, 1.666, 0.002, w)
                + 0.55 * narrow(r, 1.75, 0.003, w) + 0.55 * narrow(r, 1.786, 0.003, w) + 0.4 * narrow(r, 1.846, 0.002, w)
                + 0.5 * narrow(r, 1.863, 0.003, w) + 0.6 * narrow(r, 1.89, 0.003, w) + 0.85 * narrow(r, 2.001, 0.008, w);
        return clamp(u, 0.0, 0.9);
    }
    if (u_ring_style == 2) {   // Neptune: faint dusty bands and two narrow rings
        float n = 0.12 * narrow(r, 1.69, 0.08, w) + 0.5 * narrow(r, 2.148, 0.004, w) + 0.1 * narrow(r, 2.3, 0.08, w)
                + 0.6 * narrow(r, 2.539, 0.005, w);
        return clamp(n, 0.0, 0.9);
    }
    float a = 0.0;
    a += 0.16 * (smoothstep(1.24 - w, 1.24 + w, r) - smoothstep(1.525 - w, 1.525 + w, r));   // C ring
    a += 0.92 * (smoothstep(1.525 - w, 1.525 + w, r) - smoothstep(1.95 - w, 1.95 + w, r));   // B ring
    a += 0.07 * (smoothstep(1.95 - w, 1.95 + w, r) - smoothstep(2.025 - w, 2.025 + w, r));   // Cassini division
    a += 0.62 * (smoothstep(2.025 - w, 2.025 + w, r) - smoothstep(2.27 - w, 2.27 + w, r));   // A ring
    a *= 1.0 - 0.8 * (smoothstep(2.21 - w, 2.212, r) - smoothstep(2.216, 2.218 + w, r));   // Encke gap
    float detail = clamp(1.0 - w * 120.0, 0.0, 1.0);
    a *= 1.0 - detail * 0.18 * (0.5 + 0.5 * sin(r * 260.0 + sin(r * 77.0) * 3.0));
    return clamp(a, 0.0, 0.97);
}

vec3 space(vec3 d) {
    float e = d.y;
    vec3 col = mix(u_horizon, u_mid, smoothstep(-0.02, 0.22, e));
    col = mix(col, u_zenith, smoothstep(0.2, 0.85, e));
    if (u_stars > 0.0) {
        vec3 q = d * 380.0;
        vec3 cell = floor(q);
        float h = hash13(cell);
        if (h > 0.985) {
            vec3 c = cell + vec3(hash13(cell + 1.7), hash13(cell + 4.1), hash13(cell + 7.3));
            float dd = length(q - c) / 380.0;
            float tw = 0.75 + 0.25 * sin(u_time * (2.0 + h * 9.0) + h * 40.0);
            float s = exp(-pow(dd / (u_px * 0.9), 2.0)) * (h - 0.985) / 0.015 * tw;
            col += vec3(0.85, 0.9, 1.0) * s * u_stars * smoothstep(0.0, 0.25, e) * 1.6;
        }
    }
    return col;
}

// Glowing gas in the black hole's disk at point q (relative to the hole, in horizon radii).
vec4 disk_at(vec3 q, vec3 ray) {
    vec3 N = u_body_frame[1];
    float r = length(q);
    if (r < u_disk.x * 0.85 || r > u_disk.y) return vec4(0.0);
    float inner = smoothstep(u_disk.x * 0.85, u_disk.x * 1.15, r);
    float outer = 1.0 - smoothstep(u_disk.y * 0.45, u_disk.y, r);
    float heat = pow(u_disk.x / r, 1.7);
    vec3 tang = normalize(cross(N, q));
    float beam = pow(clamp(1.0 + 0.6 * dot(tang, -ray), 0.3, 1.7), 3.0);   // the side spinning toward us is brighter
    vec3 X = u_body_frame[0];
    float ang = atan(dot(q, cross(N, X)), dot(q, X));
    float swirl = 0.72 + 0.28 * sin(ang * 7.0 - log(r) * 16.0 - u_time * 1.2) * sin(ang * 3.0 + log(r) * 6.0 + 1.3);
    vec3 c = mix(vec3(1.0, 0.9, 0.74), vec3(1.0, 0.4, 0.1), smoothstep(u_disk.x, u_disk.y * 0.6, r));
    float a = clamp(inner * outer * (0.3 + heat) * swirl, 0.0, 0.92);
    return vec4(c * heat * beam * swirl * u_disk_bright, a);
}

void main() {
    vec4 p = u_inv_viewproj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - u_cam);
    float e = dir.y;
    vec3 col = space(dir);

    if (u_milky > 0.0) {   // the Milky Way: a lumpy band with dark dust lanes, brightest toward the center
        vec3 Ng = normalize(vec3(0.459, -0.831, -0.315));   // arcs across the frame, low on the left to high on the right
        vec3 Gc = normalize(cross(Ng, vec3(1.0, 0.0, 0.0)));
        vec3 Gb = cross(Ng, Gc);
        float glat = asin(clamp(dot(dir, Ng), -1.0, 1.0));
        float glon = atan(dot(dir, Gb), dot(dir, Gc));
        float toward = 0.5 + 0.5 * cos(glon);
        vec2 gp = vec2(dot(dir, Gc), dot(dir, Gb));   // seamless coordinates along the band
        float n = fbm(gp * 7.0 + vec2(glat * 9.0, -glat * 6.0));
        float band = exp(-pow(glat / (0.055 + 0.05 * toward), 2.0)) * (0.35 + 1.0 * n) * (0.4 + 0.9 * toward);
        float lane = smoothstep(0.5, 0.7, fbm(gp * 18.0 + vec2(3.0, glat * 40.0))) * exp(-pow(glat / 0.022, 2.0));
        col += vec3(0.86, 0.86, 0.95) * band * (1.0 - 0.8 * lane) * 0.16 * u_milky * smoothstep(-0.02, 0.1, e);
        vec3 Am = normalize(vec3(-0.36, 0.45, -0.82));   // Andromeda: a faint smudge
        vec3 Ar = normalize(cross(Am, vec3(0.0, 1.0, 0.0)));
        vec2 aq = vec2(dot(dir, Ar), dot(dir, cross(Ar, Am))) / 0.012;
        aq = mat2(0.8, -0.6, 0.6, 0.8) * aq;
        col += vec3(0.85, 0.82, 0.78) * 0.5 * u_milky * exp(-(aq.x * aq.x / 4.0 + aq.y * aq.y)) * step(0.0, dot(dir, Am));
    }
    if (u_galaxy > 0.0) {   // our galaxy from outside: a tilted spiral with a golden core
        vec3 C = u_galaxy_dir;
        vec3 Rg = normalize(cross(C, vec3(0.0, 1.0, 0.0)));
        vec3 Ug = cross(Rg, C);
        float fwd = dot(dir, C);
        vec2 q = vec2(dot(dir, Rg), dot(dir, Ug)) / max(fwd, 1e-3) / tan(u_galaxy_r);
        q = mat2(0.94, 0.34, -0.34, 0.94) * q;
        q.y /= 0.58;
        float r = length(q);
        float a = atan(q.y, q.x);
        float swirl = a - 2.4 * log(max(r, 1e-3));
        float arms = pow(0.5 + 0.5 * cos(2.0 * swirl), 4.0);
        float lanes = smoothstep(0.55, 0.9, pow(0.5 + 0.5 * cos(2.0 * swirl + 0.7), 2.0));
        float disk = exp(-r / 0.3) * (1.0 - smoothstep(0.85, 1.15, r));
        float n = fbm(q * 10.0 + 2.0);
        float bulge = exp(-pow(r / 0.11, 2.0));
        float stars = smoothstep(0.62, 0.8, fbm(q * 40.0 + 7.0)) * disk * arms;   // knots of young stars
        vec3 g = vec3(0.62, 0.7, 1.0) * disk * (0.12 + 1.0 * arms) * (0.6 + 0.7 * n) * (1.0 - 0.6 * lanes * disk);
        g += vec3(0.8, 0.85, 1.0) * stars * 0.8 + vec3(1.0, 0.85, 0.6) * bulge * 1.1;
        col += g * 0.5 * u_galaxy * step(0.0, fwd);
    }
    if (u_sun_show > 0.0) {   // our Sun: a disk that shrinks to a point
        float th = acos(clamp(dot(dir, u_sun_dir), -1.0, 1.0));
        float R = max(u_sun_r, 1e-7);
        float disk = (1.0 - smoothstep(R - u_px, R + u_px, th)) * step(1.2 * u_px, R);
        col = mix(col, vec3(1.0, 0.96, 0.88) * u_sun_b, disk);
        col += vec3(1.0, 0.94, 0.84) * min(u_sun_b, 6.0) * 0.5 * exp(-pow(th / (1.5 * u_px), 2.0));
        col += vec3(1.0, 0.86, 0.62) * u_sun_glow * exp(-th / 0.12);
    }

    if (u_body_type == 1) {
        // Black hole: a shadow, the sky bent around it, and a disk of hot gas (front and lensed back images).
        vec3 S = u_body_dir;
        float D = max(u_body_d, 1.0005);
        float cosT = dot(dir, S);
        float theta = acos(clamp(cosT, -1.0, 1.0));
        float g = sqrt(max(1.0 - 1.0 / D, 1e-4));
        float sp = min(2.598 / D * g, 1.0);
        float psi = D >= 1.5 ? asin(sp) : 3.14159265 - asin(sp);
        float bimp = theta < 1.5707963 ? D * sin(theta) / g : 1e3;
        float x = max(bimp - 2.598, 1e-3);
        float alpha = min(2.0 / bimp + 2.945 / (bimp * bimp) + 0.35 * log(1.0 + 0.6 / x), 3.0);
        vec3 perp = normalize(S - dir * cosT + vec3(1e-7));
        vec3 bent = normalize(dir * cos(alpha) + perp * sin(alpha));
        float outside = smoothstep(psi - 1.2 * u_px, psi + 1.2 * u_px, theta);
        vec3 lensed = mix(col, space(bent), clamp(alpha * 5.0, 0.0, 1.0));
        col = mix(col * u_bh_veil, lensed, outside);
        vec3 C = S * D;
        float tca = D * cosT;
        vec3 N = u_body_frame[1];
        if (outside > 0.0 && tca > 0.0) {   // back of the disk, bent up over the shadow
            vec3 P = dir * tca;
            float dn2 = dot(bent, N);
            if (abs(dn2) > 1e-6) {
                float t2 = dot(C - P, N) / dn2;
                if (t2 > 0.0) {
                    vec4 dk = disk_at(P + bent * t2 - C, bent);
                    col = mix(col, dk.rgb, dk.a * outside);
                }
            }
        }
        float dn = dot(dir, N);
        if (abs(dn) > 1e-6) {               // front of the disk, straight to us
            float t1 = dot(C, N) / dn;
            if (t1 > 0.0 && (t1 < tca || outside < 1.0)) {
                vec4 dk = disk_at(dir * t1 - C, dir);
                col = mix(col, dk.rgb, dk.a);
            }
        }
        // Photon ring, glow in our air around it, and the faint point it starts as.
        float ringw = max(psi * 0.035, 1.5 * u_px);
        col += vec3(1.0, 0.62, 0.3) * 1.4 * u_disk_bright * exp(-pow((theta - psi) / ringw, 2.0)) * step(1.5, D);
        col += u_glow_col * u_glow * 0.35 * exp(-max(theta - psi, 0.0) / (psi * 1.2 + 0.03)) * outside;
        float point = 1.0 - smoothstep(1.5 * u_px, 4.0 * u_px, psi);
        col += vec3(1.0, 0.7, 0.45) * 2.4 * point * exp(-pow(theta / (1.4 * u_px), 2.0));
    }

    if (u_body_type == 2) {
        vec3 S = u_body_dir;
        float th = acos(clamp(dot(dir, S), -1.0, 1.0));
        // Tails: along the great circle from the head toward u_tail_dir; dust curves and widens, ions stay straight.
        vec3 Tt = normalize(u_tail_dir - S * dot(u_tail_dir, S));
        vec3 Bt = cross(S, Tt);
        float ax = atan(dot(dir, Tt), dot(dir, S));
        float ay = asin(clamp(dot(dir, Bt), -1.0, 1.0));
        float along = max(ax, 0.0);
        float start = smoothstep(-0.004, 0.02, ax) * step(dot(dir, S), 1.0) * step(0.0, dot(dir, S) + 0.3);
        float curve = 0.35 * along * along;
        float w = 0.01 + along * 0.32;
        float tail_on = step(1e-4, u_tail_len);  // asteroids have no tail (and no division by zero)
        float tl = max(u_tail_len, 1e-4);
        float dust = exp(-pow((ay - curve) / w, 2.0)) * exp(-along / tl) * start * tail_on;
        float ion = exp(-pow(ay / (0.004 + along * 0.025), 2.0)) * exp(-along / (tl * 1.5)) * start * tail_on;
        float streaks = 0.9 + 0.1 * sin(ay / w * 6.0 + along * 18.0 - u_time * 0.3);
        float headward = 1.0 + 1.5 * exp(-along / 0.06);   // brightest close to the head
        vec3 tails = (u_tail_col * dust * streaks * 0.8 + vec3(0.5, 0.72, 1.0) * ion * 0.5) * headward;
        col += tails * u_comet_bright * (1.0 - u_entry);
        float cr = max(u_coma_r, 0.006);
        float big = 1.0 - 0.75 * smoothstep(0.03, 0.2, cr);   // a coma filling the sky is a faint haze, not a glare
        col += vec3(0.82, 0.93, 1.0) * exp(-th / cr) * u_comet_bright * 1.2 * big * (1.0 - u_entry);
        col += vec3(1.0, 0.98, 0.94) * 3.0 * exp(-pow(th / max(cr * 0.18, 1.6 * u_px), 2.0)) * u_comet_bright;
        if (u_entry > 0.0) {   // burning through the air: a white-hot head dragging smoke
            vec3 Tr = normalize(u_trail_dir - S * dot(u_trail_dir, S));
            float tx = atan(dot(dir, Tr), dot(dir, S));
            float ty = asin(clamp(dot(dir, cross(S, Tr)), -1.0, 1.0));
            float on = smoothstep(-0.003, 0.01, tx) * (1.0 - smoothstep(u_trail_len * 0.85, u_trail_len, tx));
            float tw = 0.009 + tx * 0.09;
            float n = fbm(vec2(tx * 70.0 - u_time * 2.0, ty * 140.0));
            float smoke = exp(-pow(ty / (tw * (0.8 + 0.5 * n)), 2.0)) * on;
            vec3 smokec = mix(vec3(1.0, 0.55, 0.2) * 2.5, vec3(0.38, 0.35, 0.33) * (0.7 + 0.5 * n), smoothstep(0.0, 0.05, tx));
            col = mix(col, smokec, clamp(smoke * (0.75 + 0.7 * n), 0.0, 0.95) * u_entry);
            col += vec3(1.0, 0.92, 0.75) * 8.0 * exp(-pow(th / (0.012 + 0.016 * u_entry), 2.0)) * u_entry;
            col += vec3(1.0, 0.62, 0.3) * 1.4 * exp(-th / 0.1) * u_entry;
        }
    }
    if (u_fireball > 0.0 || u_plume > 0.0) {
        // Where it landed, beyond the peaks: a fireball dome, then a mushroom of smoke climbing the sky.
        vec3 G = u_site_dir;
        vec3 R = normalize(cross(G, vec3(0.0, 1.0, 0.0)));
        float hx = asin(clamp(dot(dir, R), -1.0, 1.0));
        float hy = asin(clamp(dir.y, -1.0, 1.0)) - asin(clamp(G.y, -1.0, 1.0));
        float d = length(vec2(hx, hy * 1.15));
        float r = max(u_fireball_r, 1e-4);
        float dome = 1.0 - smoothstep(r * 0.75, r, d);
        vec3 hot = mix(vec3(1.0, 0.95, 0.85) * 4.0, vec3(1.0, 0.45, 0.12) * 1.6, clamp(1.0 - u_fireball, 0.0, 1.0));
        col = mix(col, hot, dome * clamp(u_fireball * 1.5, 0.0, 1.0));
        col += vec3(1.0, 0.6, 0.3) * u_fireball * 0.9 * exp(-d / (r * 2.5 + 0.02));
        if (u_plume > 0.0) {
            float top = 0.5 * u_plume;
            float capw = 0.06 + 0.26 * u_plume;
            float colw = 0.02 + 0.05 * u_plume + max(hy, 0.0) * 0.12;
            float stem = step(0.0, hy) * step(hy, top) * (1.0 - smoothstep(colw * 0.7, colw, abs(hx)));
            vec2 c = vec2(hx / capw, (hy - top) / (capw * 0.45));
            float n = fbm(vec2(hx * 18.0, hy * 18.0 - u_time * 0.15) + 3.0);
            float cap = 1.0 - smoothstep(0.75, 1.0 + 0.25 * n, length(c));
            float body = clamp(max(stem, cap) * (0.65 + 0.6 * n), 0.0, 1.0);
            float lower = 1.0 - smoothstep(0.0, top + 0.05, hy);
            vec3 smoke = mix(u_plume_lit, vec3(0.22, 0.19, 0.17), 0.6) * (0.55 + 0.7 * n);
            smoke = mix(smoke, vec3(1.0, 0.48, 0.16) * 1.6, lower * 0.7 * u_fireball);
            col = mix(col, smoke, clamp(body * 1.2, 0.0, 1.0));
        }
    }

    // Moon.
    float mcos = dot(dir, u_moon_dir);
    float mang = acos(clamp(mcos, -1.0, 1.0));
    col += vec3(0.55, 0.62, 0.8) * 0.12 * exp(-mang / 0.06) * u_moon_vis;
    if (mang < u_moon_r * 1.5) {
        vec3 right = normalize(cross(u_moon_dir, vec3(0, 1, 0)));
        vec3 up = cross(right, u_moon_dir);
        vec2 o = vec2(dot(dir, right), dot(dir, up)) / u_moon_r;
        float rr = length(o);
        if (rr < 1.0) {
            vec3 nrm = o.x * right + o.y * up + sqrt(1.0 - rr * rr) * (-u_moon_dir);
            float lit = max(dot(nrm, u_sun_dir), 0.0);
            float maria = smoothstep(0.45, 0.75, hnoise2(o * 2.3 + 5.0)) * 0.35 + hnoise2(o * 9.0) * 0.12;
            vec3 m = vec3(0.92, 0.9, 0.84) * (1.0 - maria) * (0.06 + 1.6 * lit);
            float cover = 1.0 - smoothstep(1.0 - 1.5 * u_px / u_moon_r, 1.0, rr);
            col = mix(col, m, cover * u_moon_vis);
        }
    }

    if (u_body_type == 3) {   // a star: a glowing disk with a darker rim, a halo, and a point when it's far
        float th = acos(clamp(dot(dir, u_body_dir), -1.0, 1.0));
        float R = asin(min(1.0 / u_body_d, 1.0));
        float edge = max(u_px, R * 0.02);
        float disk = 1.0 - smoothstep(R - edge, R + edge, th);
        float mu = sqrt(max(1.0 - pow(th / max(R, 1e-7), 2.0), 0.0));
        vec3 dir2 = normalize(dir - u_body_dir * dot(dir, u_body_dir));
        float spots = 0.85 + 0.15 * hnoise2(vec2(atan(dir2.y, dir2.x) * 3.0, th / max(R, 1e-6) * 4.0) + u_time * 0.05);
        col = mix(col, u_star_col * u_star_bright * (0.45 + 0.55 * mu) * spots, disk);
        col += u_star_col * u_star_bright * 0.22 * exp(-max(th - R, 0.0) / (R * 0.7 + 0.004)) * (1.0 - disk);
        col += u_star_col * 0.3 * u_glow * exp(-th / 0.35);   // its light scattering in our air, once it's bright
        float point = 1.0 - smoothstep(1.5 * u_px, 4.0 * u_px, R);
        col += u_star_col * 3.0 * point * exp(-pow(th / (1.4 * u_px), 2.0));
    }
    if (u_body_type == 4) {   // a pulsar: a tiny blinding point, two beams sweeping like a lighthouse
        vec3 S = u_body_dir;
        float cs = dot(dir, S);
        float th = acos(clamp(cs, -1.0, 1.0));
        vec3 v = dir - S * cs;   // offset from the pulsar in the sky plane (~ angle, near it)
        for (int k = 0; k < 2; k++) {
            vec3 B = (k == 0 ? 1.0 : -1.0) * u_beam_dir;
            vec3 Bp = B - S * dot(B, S);
            float pl = length(Bp);
            if (pl < 1e-4 || cs < 0.0) continue;
            vec3 Bn = Bp / pl;
            float along = dot(v, Bn);
            float across = length(v - Bn * along);
            float len = max(u_beam_len * (0.15 + 0.85 * pl), 1e-4);   // foreshortened as it turns toward or away from us
            float w = max(0.0018 + along * 0.045, 1.1 * u_px);       // a cone: wider farther out
            float a = exp(-pow(across / w, 2.0)) * smoothstep(-0.001, 0.006, along) * exp(-max(along, 0.0) / len);
            float core = exp(-pow(across / (w * 0.3), 2.0));          // a hot thread down the middle
            float toward = 0.55 + 0.45 * max(-dot(B, S), 0.0);
            col += u_beam_col * u_beam_bright * toward * (0.55 * a + 0.6 * core * a);
        }
        float R = asin(min(1.0 / u_body_d, 1.0));
        float disk = (1.0 - smoothstep(R - u_px, R + u_px, th)) * step(1.5 * u_px, R);
        col = mix(col, u_star_col * u_star_bright * 1.4, disk);
        col += u_star_col * u_star_bright * exp(-pow(max(th - R, 0.0) / (1.4 * u_px), 2.0));
        col += u_star_col * u_star_bright * 0.18 * exp(-max(th - R, 0.0) / (7.0 * u_px));
        // The lighthouse flash: a burst around the point when a beam sweeps across us.
        col += u_beam_col * u_pulse * (4.0 * exp(-pow(th / (3.0 * u_px), 2.0)) + 0.9 * exp(-th / (0.012 + 0.03 * u_glow))
                                       + 0.12 * u_glow * exp(-th / 0.45));
        col += u_star_col * 0.14 * u_glow * exp(-th / 0.35);   // its light scattering in our air
    }
    if (u_body_type == 0) {
    // The body: ray-traced oblate sphere and ring plane, in its own frame.
    mat3 Ft = transpose(u_body_frame);
    vec3 o = Ft * (-u_body_dir * u_body_d);   // camera, relative to the body's center
    vec3 r = Ft * dir;
    vec3 os = vec3(o.x, o.y / u_flat, o.z);
    vec3 rs = vec3(r.x, r.y / u_flat, r.z);
    float rl = length(rs);
    vec3 rn = rs / rl;
    float b = -dot(os, rn);
    float hdist = length(cross(rn, os));
    float edge = max(u_px * u_body_d, 1e-5);
    float tP = 1e30;
    vec3 planet = vec3(0.0);
    float cover = 0.0;
    vec3 Ls = normalize(Ft * u_sun_dir);
    if (b > 0.0 && hdist < 1.0 + edge) {
        float hh = min(hdist, 0.99999);
        float ts = b - sqrt(1.0 - hh * hh);
        tP = ts / rl;
        vec3 P = os + rn * ts;               // on the unit sphere
        vec3 N = normalize(vec3(P.x, P.y / u_flat, P.z));
        vec3 Pt = vec3(P.x, P.y * u_flat, P.z);
        float lon = atan(P.z, P.x);
        float near = clamp(1.0 - u_body_d / 8.0, 0.0, 1.0);   // close up: show the storms in the bands
        float lat = P.y + near * 0.012 * (fbm(vec2(lon * 24.0, P.y * 90.0)) - 0.5);
        float w = sin(lat * 24.0 + sin(lat * 9.0) * 1.4 + 0.25 * sin(lon * 3.0 + lat * 30.0));
        vec3 base = mix(u_band_a, u_band_b, 0.5 + 0.5 * w);
        base *= 0.93 + 0.07 * sin(lat * 95.0 + sin(lat * 33.0) * 2.0 + hnoise2(vec2(lon * 6.0, lat * 40.0)) * 2.0);
        base *= 1.0 + near * 0.25 * (fbm(vec2(lon * 60.0, lat * 260.0)) - 0.5);
        if (u_spot.z > 0.0) {   // a great storm: an oval with swirls and a pale collar
            vec2 q = vec2((lon - u_spot.y) / (u_spot.z * 2.2), (lat - u_spot.x) / u_spot.z);
            float r = length(q);
            float swirl = 0.85 + 0.15 * sin(atan(q.y, q.x) * 3.0 + r * 12.0);
            base = mix(base, base * 1.12 + 0.04, (1.0 - smoothstep(1.0, 1.45, r)) * 0.5);
            base = mix(base, u_spot_col * swirl, 1.0 - smoothstep(0.7, 1.0, r));
        }
        if (u_surface > 0.5) {   // rocky: dark albedo regions and fine terrain, no bands
            vec2 q = P.xy;   // the visible hemisphere seen face-on: no seam
            float big = fbm(q * 2.6 + vec2(3.7, 1.3));
            float mid = fbm(q * 9.0 + vec2(11.2, 5.1));
            float fine = fbm(q * 34.0 + vec2(2.3, 8.8));
            float dark = smoothstep(0.42, 0.62, big + 0.18 * (mid - 0.47));
            base = mix(u_band_a, u_band_b, dark);
            base *= 0.86 + 0.28 * mid;
            base *= 1.0 + (0.12 + 0.25 * near) * (fine - 0.47);
            base = mix(base, u_band_c, smoothstep(0.86, 0.9, abs(P.y) + 0.05 * (mid - 0.47)));
        } else {
            base = mix(base, u_band_c, smoothstep(0.72, 0.95, abs(lat)));
        }
        float lambert = max(dot(N, Ls), 0.0);
        float mu = max(dot(N, -r), 0.0);
        float shade = lambert * (0.6 + 0.4 * pow(mu, 0.35));
        if (u_rings.y > 0.0 && abs(Ls.y) > 1e-4) {   // ring shadow on the planet
            float tr = -Pt.y / Ls.y;
            if (tr > 0.0) {
                vec3 Q = Pt + Ls * tr;
                shade *= 1.0 - 0.85 * ring_alpha(length(Q.xz), 0.01);
            }
        }
        planet = base * shade * u_body_bright;
        cover = 1.0 - smoothstep(1.0 - edge, 1.0 + edge, hdist);
        float limb = exp(-max(1.0 - hdist, 0.0) * 18.0);
        planet += u_glow_col * limb * 0.25 * u_body_bright * (0.3 + lambert);
    }
    // Atmosphere glow around the limb, stronger as it gets close.
    if (b > 0.0) {
        float outside = max(hdist - 1.0, 0.0);
        col += u_glow_col * (0.25 + u_glow) * 0.35 * exp(-outside * 14.0) * u_body_bright;
        col += u_glow_col * u_glow * 0.12 * exp(-outside * 2.5);
    }

    vec3 ringc = vec3(0.0);
    float ra = 0.0;
    float tR = 1e30;
    if (u_rings.y > 0.0 && abs(r.y) > 1e-6) {
        tR = -o.y / r.y;
        if (tR > 0.0) {
            vec3 Q = o + r * tR;
            float rad = length(Q.xz);
            float w = max(fwidth(rad), 0.0015);
            ra = ring_alpha(rad, w);
            if (ra > 0.0) {
                float lit = 0.35 + 0.65 * abs(Ls.y) * 3.0;
                vec3 Qs = vec3(Q.x, Q.y / u_flat, Q.z);
                vec3 Lss = normalize(vec3(Ls.x, Ls.y / u_flat, Ls.z));
                float shadow = (length(cross(Lss, Qs)) < 1.0 && dot(Qs, Lss) < 0.0) ? 0.08 : 1.0;
                float tone = 0.85 + 0.15 * sin(rad * 31.0);
                ringc = u_ring_col * tone * min(lit, 1.4) * shadow * u_body_bright;
            }
        } else {
            tR = 1e30;
        }
    }

    // Too small to resolve: a bright point, like a planet in the night sky.
    float ang = acos(clamp(dot(dir, u_body_dir), -1.0, 1.0));
    float apparent = 1.0 / u_body_d;
    float point = 1.0 - smoothstep(1.5 * u_px, 4.0 * u_px, apparent);
    col += vec3(1.0, 0.92, 0.75) * 2.6 * point * exp(-pow(ang / (1.3 * u_px), 2.0));
    col += vec3(1.0, 0.9, 0.7) * 0.25 * point * exp(-ang / (6.0 * u_px));
    cover *= 1.0 - point;
    ra *= 1.0 - point;

    if (tR < tP) {
        col = mix(col, planet, cover);
        col = mix(col, ringc, ra);
    } else {
        col = mix(col, ringc, ra);
        col = mix(col, planet, cover);
    }

    }

    // Clouds: a thin layer in our own sky, so they drift in front of everything up there.
    if (e > 0.01) {
        vec2 uv = dir.xz / (e + 0.06) * 0.8 + vec2(u_time * 0.01, u_time * 0.004);
        float d = fbm(uv * 1.1);
        float c = smoothstep(0.5, 0.78, d) * smoothstep(0.01, 0.1, e) * (1.0 - 0.6 * smoothstep(0.5, 0.9, e));
        col = mix(col, u_cloud * (0.85 + 0.3 * d), c * 0.6 * u_cloud_amt);
    }
    col = mix(col, dust_color(u_cam_world, dir), dust_cover(u_cam_world, dir));
    f_color = vec4(col, 1.0);
}
"""

BRIGHT = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform float u_threshold;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = vec3(0.0);
    for (int i = -1; i <= 1; i += 2)
        for (int j = -1; j <= 1; j += 2)
            c += texture(u_tex, v_uv + vec2(i, j) * u_texel).rgb;
    c *= 0.25;
    float l = max(max(c.r, c.g), c.b);
    f_color = vec4(c * max(l - u_threshold, 0.0) / max(l, 1e-4), 1.0);
}
"""

BLUR = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_step;
in vec2 v_uv;
out vec4 f_color;
void main() {
    float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 c = texture(u_tex, v_uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(u_tex, v_uv + u_step * float(i)).rgb * w[i];
        c += texture(u_tex, v_uv - u_step * float(i)).rgb * w[i];
    }
    f_color = vec4(c, 1.0);
}
"""

COMPOSITE = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom1;
uniform sampler2D u_bloom2;
uniform float u_exposure;
uniform float u_flash;
uniform vec3 u_flash_col;
uniform float u_fade;
uniform float u_time;
uniform vec2 u_size;
uniform float u_grade;   // 0 = neutral; 1 = a gentle contrast curve and a little more color
in vec2 v_uv;
out vec4 f_color;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    c += texture(u_bloom1, v_uv).rgb * 0.55 + texture(u_bloom2, v_uv).rgb * 0.45;
    c = aces(c * u_exposure);
    vec2 d = (v_uv - 0.5) * vec2(1.0, 0.75);
    c *= mix(0.72, 1.0, smoothstep(0.75, 0.2, length(d)));
    c = pow(c, vec3(1.0 / 2.2));
    if (u_grade > 0.0) {
        vec3 s = c * c * (3.0 - 2.0 * c);                       // S-curve
        c = mix(c, s, 0.35 * u_grade);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = clamp(mix(vec3(l), c, 1.0 + 0.12 * u_grade), 0.0, 1.0);  // saturation
    }
    c = mix(c, u_flash_col, u_flash);
    c *= 1.0 - u_fade;
    c += (hash12(v_uv * u_size + fract(u_time) * 97.0) - 0.5) / 255.0;
    f_color = vec4(c, 1.0);
}
"""


@dataclass
class Frame:
    """Everything one frame needs, besides the instances."""

    cam_pos: np.ndarray
    cam_look: np.ndarray
    fov: float
    time: float
    zenith: tuple
    mid: tuple
    horizon: tuple
    stars: float
    moon_dir: np.ndarray
    moon_r: float
    moon_col: tuple
    sun_dir: np.ndarray
    body_dir: np.ndarray
    body_d: float
    body_frame: np.ndarray  # 3x3, columns
    flattening: float
    rings: tuple
    bands: tuple
    ring_col: tuple
    body_bright: float
    body_light: tuple
    glow_col: tuple
    glow: float
    amb_sky: tuple
    amb_ground: tuple
    fog_col: tuple
    fog_density: float
    power: float
    lava: float
    exposure: float
    flash: float
    flash_col: tuple
    fade: float
    lights: list = None
    cloud: tuple = (0.12, 0.13, 0.25)
    body_type: int = 0
    disk: tuple = (3.0, 9.0)
    disk_bright: float = 1.0
    bh_veil: float = 0.15
    moon_vis: float = 1.0
    key_dir: np.ndarray | None = None  # the scene's main light; defaults to the moon
    key_col: tuple | None = None
    shadow_vp: np.ndarray | None = None  # light view-projection for the shadow map
    snow_line: float = 1e9
    ring_style: int = 0
    star_col: tuple = (1.0, 0.45, 0.2)
    star_bright: float = 1.0
    spot: tuple = (0.0, 0.0, 0.0)
    surface: float = 0.0
    sun_show: float = 0.0
    sun_r: float = 0.0
    sun_b: float = 0.0
    sun_glow: float = 0.0
    milky: float = 0.0
    galaxy: float = 0.0
    galaxy_r: float = 0.6
    galaxy_dir: tuple = (0.0, 0.3, -0.95)
    frost: float = 0.0
    ice: float = 0.0
    dry: float = 0.0  # heat: plants dry out
    scorch: float = 0.0  # heat: everything chars
    boil: float = 0.0  # heat: water boils
    cloud_amt: float = 1.0
    spot_col: tuple = (0.75, 0.35, 0.2)
    sky_extra: dict | None = None  # extra sky uniforms, e.g. the comet's
    dust_front: float | None = None  # z of an advancing dust wall
    dust_h: float = 600.0
    dust_col: tuple = (0.5, 0.3, 0.18)


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _gl(m: np.ndarray) -> bytes:
    return np.asarray(m, dtype="f4").T.tobytes()  # column-major


def ortho(half: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = m[1, 1] = 1.0 / half
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


SHADOW_VERTEX = """
#version 330
uniform mat4 u_lightvp;
in vec3 in_pos;
in mat4 in_model;
void main() {
    gl_Position = u_lightvp * in_model * vec4(in_pos, 1.0);
}
"""

SHADOW_FRAGMENT = """
#version 330
void main() {}
"""


def create_context() -> moderngl.Context:
    try:
        return moderngl.create_standalone_context(backend="egl")
    except Exception:
        return moderngl.create_standalone_context()


class Renderer:
    def __init__(self, width: int, height: int, samples: int = 4, scale: int = 1, shadow: int = 2048,
                 grade: float = 0.0):
        """width x height is the output; the scene is drawn at scale x that and filtered down (supersampling)."""
        self.out_w, self.out_h = width, height
        self.w, self.h = width * scale, height * scale
        self.shadow_size = shadow
        self.grade = grade
        ctx = self.ctx = create_context()
        self.scene_prog = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.sky_prog = ctx.program(vertex_shader=FULLSCREEN.replace("vec4(in_pos, 0.0, 1.0)", "vec4(in_pos, 1.0, 1.0)"),
                                    fragment_shader=SKY)
        self.bright_prog = ctx.program(vertex_shader=FULLSCREEN, fragment_shader=BRIGHT)
        self.blur_prog = ctx.program(vertex_shader=FULLSCREEN, fragment_shader=BLUR)
        self.comp_prog = ctx.program(vertex_shader=FULLSCREEN, fragment_shader=COMPOSITE)
        tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.quads = {name: ctx.vertex_array(p, [(tri, "2f", "in_pos")])
                      for name, p in (("sky", self.sky_prog), ("bright", self.bright_prog),
                                      ("blur", self.blur_prog), ("comp", self.comp_prog))}

        width, height = self.w, self.h
        self.msaa = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((width, height), 4, samples=samples, dtype="f2")],
            depth_attachment=ctx.depth_renderbuffer((width, height), samples=samples),
        )
        self.scene_tex = ctx.texture((width, height), 4, dtype="f2")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)  # a 2x2 box when filtering a 2x render down
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex])
        self.bloom = []
        for div in (4, 8):  # sized from the output, so the glow looks the same at any supersampling
            size = (max(self.out_w // div, 1), max(self.out_h // div, 1))
            a, b = ctx.texture(size, 4, dtype="f2"), ctx.texture(size, 4, dtype="f2")
            for t in (a, b):
                t.repeat_x = t.repeat_y = False
            self.bloom.append((size, a, b, ctx.framebuffer([a]), ctx.framebuffer([b])))
        self.out_tex = ctx.texture((self.out_w, self.out_h), 3)
        self.out_fbo = ctx.framebuffer(color_attachments=[self.out_tex])

        self.shadow_prog = ctx.program(vertex_shader=SHADOW_VERTEX, fragment_shader=SHADOW_FRAGMENT)
        self.shadow_tex = ctx.depth_texture((shadow, shadow))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.meshes = {name: ctx.buffer(make()) for name, make in PROTOTYPES.items()}
        self.counts = {name: len(make()) for name, make in PROTOTYPES.items()}
        self.static: dict[str, tuple] = {}
        self._shadow_vp = None
        self.dynamic: dict[str, list] = {}

    def add_mesh(self, name: str, triangles: np.ndarray) -> None:
        """Register a scene's own mesh (e.g. its terrain) alongside the built-in ones."""
        data = np.ascontiguousarray(triangles, dtype="f4")
        self.meshes[name] = self.ctx.buffer(data.tobytes())
        self.counts[name] = len(data)

    def _vao(self, mesh: str, buf: moderngl.Buffer) -> moderngl.VertexArray:
        return self.ctx.vertex_array(self.scene_prog, [
            (self.meshes[mesh], "3f", "in_pos"),
            (buf, "16f 4f 4f/i", "in_model", "in_color", "in_params"),
        ])

    def set_static(self, arrays: dict[str, np.ndarray]) -> None:
        for vao, buf, _ in self.static.values():
            vao.release()
            buf.release()
        self.static = {}
        self.shadow_vaos = []
        for mesh, data in arrays.items():
            buf = self.ctx.buffer(data.tobytes())
            self.static[mesh] = (self._vao(mesh, buf), buf, len(data))
            svao = self.ctx.vertex_array(self.shadow_prog, [
                (self.meshes[mesh], "3f", "in_pos"), (buf, "16f 32x/i", "in_model")])
            self.shadow_vaos.append((svao, len(data)))

    def _draw_dynamic(self, arrays: dict[str, np.ndarray]) -> None:
        for mesh, data in arrays.items():
            entry = self.dynamic.get(mesh)
            if entry is None or entry[1].size < data.nbytes:
                if entry:
                    entry[0].release()
                    entry[1].release()
                buf = self.ctx.buffer(reserve=max(data.nbytes * 2, 4096), dynamic=True)
                entry = self.dynamic[mesh] = [self._vao(mesh, buf), buf]
            entry[1].write(data.tobytes())
            entry[0].render(moderngl.TRIANGLES, vertices=self.counts[mesh], instances=len(data))

    def render(self, f: Frame, dynamic: dict[str, np.ndarray]) -> np.ndarray:
        ctx = self.ctx
        view = look_at(f.cam_pos, f.cam_look)
        proj = perspective(f.fov, self.w / self.h, 0.3, 30000.0)
        viewproj = proj @ view
        # The sky only needs directions, so it uses the view without translation.
        rot = view.copy()
        rot[:3, 3] = 0
        inv_sky = np.linalg.inv(proj @ rot)

        if f.shadow_vp is not None and not (self._shadow_vp is not None and np.allclose(f.shadow_vp, self._shadow_vp)):
            self._shadow_vp = f.shadow_vp.copy()  # depth from the sun's point of view (reused while the sun stays put)
            self.shadow_fbo.use()
            ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
            ctx.clear(depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST)
            self.shadow_prog["u_lightvp"].write(_gl(f.shadow_vp))
            for svao, n in self.shadow_vaos:
                svao.render(moderngl.TRIANGLES, instances=n)

        self.msaa.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.clear(0, 0, 0, 1, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        prog = self.scene_prog
        self.shadow_tex.use(3)
        prog["u_shadow_map"].value = 3
        prog["u_use_shadow"].value = int(f.shadow_vp is not None)
        if f.shadow_vp is not None:
            prog["u_shadow_vp"].write(_gl(f.shadow_vp))
        if "u_snow" in prog:
            prog["u_snow"].value = f.snow_line
        for name, value in (("u_frost", f.frost), ("u_ice", f.ice), ("u_dry", f.dry), ("u_scorch", f.scorch),
                            ("u_boil", f.boil)):
            if name in prog:
                prog[name].value = value
        for p_ in (prog, self.sky_prog):
            if "u_dust_on" in p_:
                p_["u_dust_on"].value = 0.0 if f.dust_front is None else 1.0
                p_["u_front_z"].value = float(f.dust_front or 0.0)
                p_["u_dust_h"].value = float(f.dust_h)
                p_["u_dust_col"].value = tuple(f.dust_col)
        if "u_cam_world" in self.sky_prog:
            self.sky_prog["u_cam_world"].value = tuple(float(v) for v in f.cam_pos)
        prog["u_viewproj"].write(_gl(viewproj))
        key_dir = f.moon_dir if f.key_dir is None else f.key_dir
        key_col = f.moon_col if f.key_col is None else f.key_col
        for name, value in (("u_cam", tuple(f.cam_pos)), ("u_time", f.time), ("u_moon_dir", tuple(key_dir)),
                            ("u_moon_col", key_col), ("u_body_dir", tuple(f.body_dir)),
                            ("u_body_col", f.body_light), ("u_amb_sky", f.amb_sky), ("u_amb_ground", f.amb_ground),
                            ("u_fog_col", f.fog_col), ("u_fog_density", f.fog_density), ("u_power", f.power),
                            ("u_lava", f.lava)):
            if name in prog:
                prog[name].value = value
        lights = np.zeros((8, 8), dtype="f4")
        for i, light in enumerate(f.lights[:8]):
            lights[i, :len(light)] = light
        if "u_lpos" in prog:
            prog["u_lpos"].write(np.ascontiguousarray(lights[:, :4]).tobytes())
            prog["u_lcol"].write(np.ascontiguousarray(np.c_[lights[:, 4:7], np.zeros(8, "f4")]).tobytes())
            prog["u_nlights"].value = min(len(f.lights), 8)
        for vao, _, n in self.static.values():
            vao.render(moderngl.TRIANGLES, instances=n)
        self._draw_dynamic(dynamic)
        # Sky last, only where nothing was drawn (it sits on the far plane).
        ctx.depth_func = "<="
        ctx.depth_mask = False
        px = math.radians(f.fov) / self.h
        sky = self.sky_prog
        sky["u_inv_viewproj"].write(_gl(inv_sky))
        sky["u_cam"].value = (0.0, 0.0, 0.0)
        for name, value in (("u_zenith", f.zenith), ("u_mid", f.mid), ("u_horizon", f.horizon),
                            ("u_stars", f.stars), ("u_px", px), ("u_sun_dir", tuple(f.sun_dir)),
                            ("u_moon_dir", tuple(f.moon_dir)), ("u_moon_r", f.moon_r),
                            ("u_body_dir", tuple(f.body_dir)), ("u_body_d", f.body_d),
                            ("u_flat", 1.0 - f.flattening), ("u_rings", f.rings), ("u_band_a", f.bands[0]),
                            ("u_band_b", f.bands[1]), ("u_band_c", f.bands[2]), ("u_ring_col", f.ring_col),
                            ("u_body_bright", f.body_bright), ("u_glow_col", f.glow_col), ("u_glow", f.glow),
                            ("u_time", f.time), ("u_cloud", f.cloud), ("u_body_type", f.body_type),
                            ("u_disk", f.disk), ("u_disk_bright", f.disk_bright), ("u_bh_veil", f.bh_veil),
                            ("u_moon_vis", f.moon_vis), ("u_ring_style", f.ring_style),
                            ("u_star_col", f.star_col), ("u_star_bright", f.star_bright),
                            ("u_spot", f.spot), ("u_spot_col", f.spot_col), ("u_sun_show", f.sun_show), ("u_surface", f.surface),
                            ("u_sun_r", f.sun_r), ("u_sun_b", f.sun_b), ("u_sun_glow", f.sun_glow),
                            ("u_milky", f.milky), ("u_galaxy", f.galaxy), ("u_galaxy_r", f.galaxy_r),
                            ("u_galaxy_dir", tuple(float(v) for v in f.galaxy_dir)), ("u_cloud_amt", f.cloud_amt)):
            if name in sky:
                sky[name].value = value
        sky["u_body_frame"].write(np.asarray(f.body_frame, dtype="f4").T.tobytes())
        for name in ("u_coma_r", "u_comet_bright", "u_tail_len", "u_entry", "u_trail_len", "u_fireball",
                     "u_fireball_r", "u_plume", "u_beam_len", "u_beam_bright", "u_pulse"):
            if name in sky:
                sky[name].value = float((f.sky_extra or {}).get(name, 0.0))
        for name in ("u_tail_dir", "u_tail_col", "u_trail_dir", "u_site_dir", "u_plume_lit", "u_beam_dir", "u_beam_col"):
            if name in sky:
                sky[name].value = tuple(float(v) for v in (f.sky_extra or {}).get(name, (0.0, 1.0, 0.0)))
        self.quads["sky"].render()

        ctx.depth_mask = True
        ctx.depth_func = "<"
        ctx.disable(moderngl.DEPTH_TEST)

        ctx.copy_framebuffer(self.scene_fbo, self.msaa)

        # Bloom: bright pass at 1/4 size, blurred, then again at 1/8 for a wide glow.
        source = self.scene_tex
        threshold = 1.15
        bloom_out = []
        for size, a, b, fa, fb in self.bloom:
            fa.use()
            ctx.viewport = (0, 0, *size)
            source.use(0)
            self.bright_prog["u_tex"].value = 0
            self.bright_prog["u_texel"].value = (1.0 / source.width, 1.0 / source.height)
            self.bright_prog["u_threshold"].value = threshold
            self.quads["bright"].render()
            for _ in range(2):
                for step, src, dst in (((1.0 / size[0], 0.0), a, fb), ((0.0, 1.0 / size[1]), b, fa)):
                    dst.use()
                    src.use(0)
                    self.blur_prog["u_tex"].value = 0
                    self.blur_prog["u_step"].value = step
                    self.quads["blur"].render()
            bloom_out.append(a)
            source, threshold = a, 0.0

        self.out_fbo.use()
        ctx.viewport = (0, 0, self.out_w, self.out_h)
        self.scene_tex.use(0)
        bloom_out[0].use(1)
        bloom_out[1].use(2)
        comp = self.comp_prog
        comp["u_scene"].value, comp["u_bloom1"].value, comp["u_bloom2"].value = 0, 1, 2
        comp["u_exposure"].value = f.exposure
        comp["u_flash"].value = f.flash
        comp["u_flash_col"].value = f.flash_col
        comp["u_fade"].value = f.fade
        comp["u_time"].value = f.time
        comp["u_size"].value = (float(self.out_w), float(self.out_h))
        comp["u_grade"].value = self.grade
        self.quads["comp"].render()
        data = self.out_fbo.read(components=3, alignment=1)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.out_h, self.out_w, 3)[::-1]

    def release(self) -> None:
        self.ctx.release()
