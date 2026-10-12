/* ============================================================
   diagram.js: tiny hand-drawn diagram renderer for project pages.
   No library to load. Exposes window.D(id, W, H, spec).
   Used by projects/project.js for ```diagram blocks in the markdown.
   ============================================================ */

/* Tiny hand-drawn diagram renderer. Boxes, cylinders and arrows are drawn as wobbly
   double-strokes so they look sketched, and there is no library to load. */
(function () {
  var FONT = "'Caveat','Segoe Print','Bradley Hand','Comic Sans MS','Chalkboard SE',cursive";
  var FILL = { blue: '#dbe8ff', green: '#d9f2dc', yellow: '#fff2c0', pink: '#ffe1e1', purple: '#eadcff',
               grey: '#f0f0f0', white: '#ffffff', orange: '#ffe5cc' };
  var NS = 'http://www.w3.org/2000/svg';

  function rng(seed) { var s = (seed >>> 0) || 1; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  function esc(t) { return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  function Draw(seed) { this.r = rng(seed); this.out = []; }
  Draw.prototype.j = function (a) { return (this.r() - 0.5) * 2 * a; };
  Draw.prototype.seg = function (x1, y1, x2, y2, amt) {
    amt = amt == null ? 1.3 : amt;
    var mx = (x1 + x2) / 2 + this.j(amt), my = (y1 + y2) / 2 + this.j(amt);
    return 'M' + (x1 + this.j(amt * 0.6)).toFixed(1) + ' ' + (y1 + this.j(amt * 0.6)).toFixed(1) +
           ' Q' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' ' + (x2 + this.j(amt * 0.6)).toFixed(1) + ' ' + (y2 + this.j(amt * 0.6)).toFixed(1);
  };
  Draw.prototype.rect = function (x, y, w, h) {
    var d = '', o = 3;
    for (var i = 0; i < 2; i++) {
      d += this.seg(x - o, y, x + w + o, y) + this.seg(x + w, y - o, x + w, y + h + o) +
           this.seg(x + w + o, y + h, x - o, y + h) + this.seg(x, y + h + o, x, y - o);
    }
    return d;
  };
  Draw.prototype.diamond = function (x, y, w, h) {
    var cx = x + w / 2, cy = y + h / 2, d = '';
    for (var i = 0; i < 2; i++) {
      d += this.seg(cx, y, x + w, cy) + this.seg(x + w, cy, cx, y + h) + this.seg(cx, y + h, x, cy) + this.seg(x, cy, cx, y);
    }
    return d;
  };
  Draw.prototype.ellipse = function (cx, cy, rx, ry, half) {
    var d = '', n = 26;
    for (var k = 0; k < 2; k++) {
      var start = this.r() * 0.4, pts = [];
      var from = half === 'bottom' ? 0 : 0, to = half === 'bottom' ? Math.PI : Math.PI * 2;
      var steps = half === 'bottom' ? n / 2 : n;
      for (var i = 0; i <= steps; i++) {
        var a = from + (to - from) * i / steps + (half === 'bottom' ? 0 : start);
        pts.push([cx + Math.cos(a) * rx + this.j(0.8), cy + Math.sin(a) * ry + this.j(0.8)]);
      }
      d += 'M' + pts.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L');
    }
    return d;
  };

  function anchor(n, side) {
    var s = side.charAt(0), f = side.length > 1 ? parseFloat(side.slice(1)) : 0.5;
    if (s === 't') return [n.x + n.w * f, n.y];
    if (s === 'b') return [n.x + n.w * f, n.y + n.h];
    if (s === 'l') return [n.x, n.y + n.h * f];
    return [n.x + n.w, n.y + n.h * f];
  }
  function horiz(side) { return side.charAt(0) === 'l' || side.charAt(0) === 'r'; }

  function route(p1, s1, p2, s2, via) {
    if (via && via.length) return [p1].concat(via, [p2]);
    var h1 = horiz(s1), h2 = horiz(s2);
    if (Math.abs(p1[0] - p2[0]) < 3 || Math.abs(p1[1] - p2[1]) < 3) return [p1, p2];
    if (h1 && h2) { var mx = (p1[0] + p2[0]) / 2; return [p1, [mx, p1[1]], [mx, p2[1]], p2]; }
    if (!h1 && !h2) { var my = (p1[1] + p2[1]) / 2; return [p1, [p1[0], my], [p2[0], my], p2]; }
    if (h1 && !h2) return [p1, [p2[0], p1[1]], p2];
    return [p1, [p1[0], p2[1]], p2];
  }

  function textLines(g, lines, cx, cy, size, weight, fill, anchorMode, halo) {
    var lh = size * 1.12, y0 = cy - (lines.length - 1) * lh / 2 + size * 0.34;
    var t = document.createElementNS(NS, 'text');
    t.setAttribute('x', cx); t.setAttribute('y', y0);
    t.setAttribute('text-anchor', anchorMode || 'middle');
    t.setAttribute('font-family', FONT); t.setAttribute('font-size', size);
    t.setAttribute('font-weight', weight || 500); t.setAttribute('fill', fill || '#1a1a1a');
    if (halo) { t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 5); t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke-linejoin', 'round'); }
    lines.forEach(function (ln, i) {
      var sp = document.createElementNS(NS, 'tspan');
      sp.setAttribute('x', cx); if (i) sp.setAttribute('dy', lh);
      sp.textContent = ln; t.appendChild(sp);
    });
    g.appendChild(t);
  }
  function path(g, d, stroke, width, fill, dash) {
    var p = document.createElementNS(NS, 'path');
    p.setAttribute('d', d); p.setAttribute('fill', fill || 'none');
    p.setAttribute('stroke', stroke || 'none'); p.setAttribute('stroke-width', width || 1.6);
    p.setAttribute('stroke-linecap', 'round'); p.setAttribute('stroke-linejoin', 'round');
    if (dash) p.setAttribute('stroke-dasharray', dash);
    g.appendChild(p); return p;
  }

  /* spec = { groups:[[x,y,w,h,label,labelXOffset]], nodes:[[id,label,x,y,w,h,color,shape]],
              edges:[[from,to,label,sideFrom,sideTo,{via,dashed,lx,ly,color}]], notes:[[x,y,text,size,anchor]] } */
  window.D = function (id, W, H, spec) {
    var host = document.getElementById(id);
    if (!host) return;
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', host.getAttribute('data-alt') || 'diagram');
    var seed = 0; for (var i = 0; i < id.length; i++) seed = seed * 31 + id.charCodeAt(i);
    var dr = new Draw(seed + 7), g = document.createElementNS(NS, 'g');
    svg.appendChild(g);

    (spec.groups || []).forEach(function (gr) {
      var x = gr[0], y = gr[1], w = gr[2], h = gr[3];
      path(g, dr.rect(x, y, w, h), '#9a9a9a', 1.3, 'none', '8 6');
      if (gr[4]) textLines(g, [gr[4]], x + 12 + (gr[5] || 0), y + 18, 17, 500, '#777', 'start');
    });

    var nodes = {};
    (spec.nodes || []).forEach(function (n) {
      var o = { id: n[0], label: n[1], x: n[2], y: n[3], w: n[4], h: n[5], color: n[6] || 'white', shape: n[7] || 'box' };
      nodes[o.id] = o;
      var fill = FILL[o.color] || o.color;
      if (o.shape === 'db') {
        var rx = o.w / 2, ry = 11, top = o.y + ry, bot = o.y + o.h - ry;
        path(g, 'M' + o.x + ' ' + top + ' L' + o.x + ' ' + bot + ' A' + rx + ' ' + ry + ' 0 0 0 ' + (o.x + o.w) + ' ' + bot +
                ' L' + (o.x + o.w) + ' ' + top + ' A' + rx + ' ' + ry + ' 0 0 0 ' + o.x + ' ' + top + ' Z', 'none', 0, fill);
        path(g, dr.seg(o.x, top, o.x, bot) + dr.seg(o.x + o.w, top, o.x + o.w, bot), '#1a1a1a', 1.7);
        path(g, dr.ellipse(o.x + rx, top, rx, ry), '#1a1a1a', 1.7);
        path(g, dr.ellipse(o.x + rx, bot, rx, ry, 'bottom'), '#1a1a1a', 1.7);
        textLines(g, o.label.split('\n'), o.x + rx, o.y + o.h / 2 + 6, 16, 500);
      } else if (o.shape === 'diamond') {
        path(g, 'M' + (o.x + o.w / 2) + ' ' + o.y + ' L' + (o.x + o.w) + ' ' + (o.y + o.h / 2) + ' L' + (o.x + o.w / 2) + ' ' + (o.y + o.h) +
                ' L' + o.x + ' ' + (o.y + o.h / 2) + ' Z', 'none', 0, fill);
        path(g, dr.diamond(o.x, o.y, o.w, o.h), '#1a1a1a', 1.7);
        textLines(g, o.label.split('\n'), o.x + o.w / 2, o.y + o.h / 2, 16, 500);
      } else {
        path(g, 'M' + (o.x + 1) + ' ' + (o.y + 1) + ' h' + (o.w - 2) + ' v' + (o.h - 2) + ' h-' + (o.w - 2) + ' Z', 'none', 0, fill);
        path(g, dr.rect(o.x, o.y, o.w, o.h), '#1a1a1a', 1.7);
        textLines(g, o.label.split('\n'), o.x + o.w / 2, o.y + o.h / 2, 16, 500);
      }
    });

    (spec.edges || []).forEach(function (e) {
      var a = nodes[e[0]], b = nodes[e[1]], opt = e[5] || {};
      if (!a || !b) { console.warn('diagram ' + id + ': missing node', e[0], e[1]); return; }
      var s1 = e[3] || 'r', s2 = e[4] || 'l';
      var pts = route(anchor(a, s1), s1, anchor(b, s2), s2, opt.via);
      var color = opt.color || '#1a1a1a', d = '', longest = 0, lp = null;
      for (var i = 0; i < pts.length - 1; i++) {
        d += dr.seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 1.1);
        var len = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
        if (len > longest) { longest = len; lp = [(pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2]; }
      }
      var last = pts[pts.length - 1], prev = pts[pts.length - 2];
      var ang = Math.atan2(last[1] - prev[1], last[0] - prev[0]), L = 11;
      [0.5, -0.5].forEach(function (da) {
        d += dr.seg(last[0], last[1], last[0] - Math.cos(ang + da) * L, last[1] - Math.sin(ang + da) * L, 0.6);
      });
      path(g, d, color, 1.6, 'none', opt.dashed ? '7 6' : null);
      if (e[2]) {
        var lx = opt.lx != null ? opt.lx : lp[0], ly = opt.ly != null ? opt.ly : lp[1];
        textLines(g, String(e[2]).split('\n'), lx, ly, 15, 500, '#444', 'middle', true);
      }
    });

    (spec.notes || []).forEach(function (n) {
      textLines(g, String(n[2]).split('\n'), n[0], n[1], n[3] || 16, 500, n[5] || '#555', n[4] || 'middle', true);
    });

    host.innerHTML = '';
    host.appendChild(svg);
  };
})();
