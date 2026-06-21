# =====================================================================
# build_skp.rb  —  SketchUp side (Path 2). Reads manifest.json from abstract.py
# and builds each face natively with its canonical Tag assigned, giving clean
# tag control that survives the fan-out to MAPP / ArrayCalc / VS / Soundvision.
#
# This is the stub. It runs; harden as you go (degenerate-face handling,
# material-per-tag, merging coplanar quads, etc).
#
# HOW TO RUN (two options):
#   A) Ruby Console (quick test): open Window > Ruby Console, then:
#        load "/full/path/build_skp.rb"
#        ScanPipe.build("/full/path/out/manifest.json")
#        Sketchup.active_model.save("/full/path/out/venue.skp")
#   B) As an extension: drop in Plugins folder; use the Extensions menu item.
#
# NOTE: SketchUp's internal unit is INCHES. Coordinates in the manifest are
# METRES, so every value is converted with .m (a SketchUp Numeric method).
# =====================================================================

require 'sketchup.rb'
require 'json'

module ScanPipe
  module_function

  # Build geometry from a manifest file into the active model.
  def build(manifest_path)
    model = Sketchup.active_model
    data  = JSON.parse(File.read(manifest_path))
    faces = data['faces'] || []

    model.start_operation('ScanPipe import', true)
    ents = model.active_entities
    made, skipped = 0, 0

    faces.each do |f|
      tag_name = f['tag'] || 'STRUCTURE'
      layer = model.layers[tag_name] || model.layers.add(tag_name)  # Tag
      pts = (f['vertices'] || []).map do |v|
        Geom::Point3d.new(v[0].to_f.m, v[1].to_f.m, v[2].to_f.m)
      end
      next (skipped += 1) if pts.length < 3
      begin
        face = ents.add_face(pts)
        if face
          face.layer = layer            # assign canonical Tag
          made += 1
        else
          skipped += 1
        end
      rescue ArgumentError              # degenerate / colinear quad
        skipped += 1
      end
    end

    model.commit_operation
    UI.messagebox("ScanPipe: built #{made} faces, skipped #{skipped}.") if defined?(UI)
    puts "ScanPipe: built #{made} faces, skipped #{skipped}."
    made
  end

  # Convenience: build then save a .skp next to the manifest.
  def build_and_save(manifest_path, skp_path)
    build(manifest_path)
    Sketchup.active_model.save(skp_path)
    puts "ScanPipe: saved #{skp_path}"
  end
end

# --- menu item (only registers once, when loaded as an extension) -----
unless defined?($scanpipe_menu_loaded)
  $scanpipe_menu_loaded = true
  if defined?(UI)
    UI.menu('Extensions').add_item('ScanPipe: import manifest…') do
      path = UI.openpanel('Choose manifest.json', '', 'JSON|*.json||')
      ScanPipe.build(path) if path
    end
  end
end
