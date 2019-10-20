import logging
import threading

import tornado.web
import tornado.gen
import tornado.websocket
import base64
import uuid
import os
from datetime import datetime
import re
import copy
import shutil
import mimetypes
import kinematics.extruder
import jinja2
from extras.gcode_macro import TemplateWrapper
import json
import io, traceback
from gcode import GCodeParser
from multiprocessing import Queue, JoinableQueue

# monkey-patch GCodeParser to emit respond events (so we can capture)

original_respond = GCodeParser.respond

def respond(self, msg):
	original_respond(self, msg)
	self.printer.send_event("gcode:response", msg)

GCodeParser.respond = respond


def parse_duration(value):
	if not value or value == -1: return value
	h_str = re.search(re.compile(r'(\d+(\s)?hours|\d+(\s)?h)'), value)
	m_str = re.search(re.compile(r'(([0-9]*\.[0-9]+)\sminutes|\d+(\s)?m)'), value)
	s_str = re.search(re.compile(r'(\d+(\s)?seconds|\d+(\s)?s)'), value)
	seconds = 0
	if h_str:
		seconds += float(max(re.findall(r'([0-9]*\.?[0-9]+)', ''.join(h_str.group())))) * 3600
	if m_str:
		seconds += float(max(re.findall(r'([0-9]*\.?[0-9]+)', ''.join(m_str.group())))) * 60
	if s_str:
		seconds += float(max(re.findall(r'([0-9]*\.?[0-9]+)', ''.join(s_str.group()))))
	if seconds == 0:
		seconds = float(max(re.findall(r'([0-9]*\.?[0-9]+)', value)))
	return seconds


class DummyGCodeParser:
	def __init__(self, gcode):
		self.gcode = gcode

	def match(self, regex, group=0):
		match = re.search(regex, self.gcode)
		if match:
			return match.group(group).strip()

	def match_last(self, regex, group=0):
		match = None
		for match in re.finditer(regex, self.gcode):
			pass

		if match:
			return match.group(group)

	def name(self):
		pass

	def height(self):
		pass

	def first_layer_height(self):
		pass

	def layer_height(self):
		pass

	def print_time(self):
		pass

	def filament(self):
		return []

	def num_layers(self):
		pass

class KISSlicerGcodeParser(DummyGCodeParser):
	def name(self):
		name = self.match(r'; (KISSlicer .+)', 1)
		if name:
			version = self.match(r'version (.+)', 1)
			return " ".join([name, version]) if version else name

	def height(self):
		return self.match_last(r'; BEGIN_LAYER_OBJECT z=(\d+.\d+)', 1)

	def layer_height(self):
		return self.match(r'; layer_thickness_mm = (\d+.\d+)', 1)

	def first_layer_height(self):
		return self.match(r'first_layer_thickness_mm = (\d+\.\d+)', 1)

	def print_time(self):
		return parse_duration(self.match(r'Estimated Build Time:\s+(.+)'))

	def filament(self):
		return self.match(r"Ext 1 =(.*)mm", 1)

class Slic3rGCodeParser(DummyGCodeParser):
	def name(self):
		return self.match(r'(Slic3r\s.*) on ', 1)

	def height(self):
		return self.match_last(r';(\d+(.\d+)?)', 1)

	def first_layer_height(self):
		return self.match(r'; first_layer_height = (\d+.\d+)', 1)

	def layer_height(self):
		return self.match(r'; layer_height = (\d+.\d+)', 1)

	def print_time(self):
		return parse_duration(self.match(r'\d+h?\s?\d+m\s\d+s'))

	def filament(self):
		return self.match(r'filament used = (\d+.\d+)mm', 1)

	def num_layer(self):
		return self.height() - self.first_layer_height() / self.layer_height() + 1


class CuraGCodeParser(DummyGCodeParser):
	def name(self):
		version = self.match(r'Cura_SteamEngine (.+)', 1)
		if version:
			return "Ultimaker Cura " + version

	def height(self):
		return self.match_last(r'\sZ(\d+.\d*)', 1)

	def first_layer_height(self):
		return self.match(r'\sZ(\d+.\d)\s', 1)

	def layer_height(self):
		return self.match(r';Layer height: (\d+.\d+)', 1)

	def print_time(self):
		return parse_duration(self.match(r'TIME:(\d+)'))

	def filament(self):
		return self.match(r';Filament used: (\d*.\d+)m', 1)

	def num_layer(self):
		return self.match(r';LAYER_COUNT:(\d)', 1)

class PrusaSlicerGCodeParser(Slic3rGCodeParser):
	def name(self):
		return self.match(r'(PrusaSlicer\s.*) on ', 1)

	def filament(self):
		return self.match(r'filament used \[mm\] = (\d+.\d+)', 1)

def get_gcode_parser(gcode):
	for cls in [PrusaSlicerGCodeParser, Slic3rGCodeParser, KISSlicerGcodeParser, CuraGCodeParser]:
		slicer = cls(gcode)
		if slicer.name():
			return slicer

	return DummyGCodeParser(gcode)


def parse_gcode(gcode):
	slicer = get_gcode_parser(gcode)
	print_time = slicer.print_time()
	height = slicer.height()
	first_layer_height = slicer.first_layer_height()
	layer_height = slicer.layer_height()
	num_layers = slicer.num_layers()

	return {
		"height": float(height) if height else None,
		"firstLayerHeight": float(first_layer_height) if first_layer_height else None,
		"layerHeight": float(layer_height) if layer_height else None,
		"printTime": int(print_time) if print_time else None,
		# "filament": slicer.filament(), # TODO: figure out why this needs an array
		"generatedBy": slicer.name(),
		"numLayers": int(num_layers) if num_layers else None
	}


class RequestHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		self.set_header("Connection", "close")
		self.set_header("Access-Control-Allow-Origin", "*")
		self.set_header("Access-Control-Allow-Methods", "*")
		self.set_header("Access-Control-Allow-Headers", "*")


class WebRootRequestHandler(RequestHandler):
	def initialize(self, manager):
		self.path = os.path.join(manager.sd_card.root_path, "www")

	@tornado.gen.coroutine
	def get(self):
		def get_content_type(path):
			mime_type, encoding = mimetypes.guess_type(path)
			if encoding == "gzip":
				return "application/gzip"
			elif encoding is not None:
				return "application/octet-stream"
			elif mime_type is not None:
				return mime_type
			else:
				return "application/octet-stream"

		path = self.path + self.request.uri

		if os.path.isfile(path):
			content_type = get_content_type(path)
			if content_type:
				self.set_header("Content-Type", content_type)

			with open(path, "rb") as f:
				self.write(f.read())
				self.finish()

		else:
			self.set_header("Content-Type", "text/html; charset=UTF-8")
			self.render(os.path.join(self.path, "index.html"))


class RestHandler(RequestHandler):
	def set_default_headers(self):
		super(RestHandler, self).set_default_headers()
		self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
		self.set_header('Content-Type', 'application/json')

	def initialize(self, manager):
		self.manager = manager

	@tornado.gen.coroutine
	def options(self, *kwargs):
		self.set_status(204)
		self.finish()

	def write_error(self, status_code, exc_info):
		out = io.BytesIO()
		traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], None, out)
		formatted = out.getvalue()
		out.close()
		self.set_header("Content-Type", "text/plain")
		self.finish(formatted)

class MachineMoveHandler(RestHandler):
	@tornado.gen.coroutine
	def post(self):
		src = self.manager.sd_card.resolve_path(self.get_argument('from'))
		dst = self.manager.sd_card.resolve_path(self.get_argument('to'))
		force = bool(self.get_argument('force'))
		if os.path.isfile(dst):
			if force:
				os.remove(dst)
			else:
				raise tornado.web.HTTPError(500, "Target file already exists")
		shutil.move(src, dst)


class MachineDirectoryHandler(RestHandler):
	@tornado.gen.coroutine
	def get(self, path):
		def file_entry(path):
			stat = os.stat(path)
			return {
				"type": "d" if os.path.isdir(path) else "f",
				"name": os.path.basename(path),
				"size": stat.st_size,
				"date": datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
			}

		real_path = self.manager.sd_card.resolve_path(path)

		files = []

		if os.path.isdir(real_path):
			files = [file_entry(real_path + "/" + p) for p in os.listdir(real_path)]

		self.finish(json.dumps(files))

	@tornado.gen.coroutine
	def put(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		if not os.path.exists(real_path):
			os.makedirs(real_path)


class MachineFileHandler(RestHandler):
	@tornado.gen.coroutine
	def put(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		dirname = os.path.dirname(real_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		with open(real_path, 'w') as file:
			file.write(self.request.body)

	@tornado.gen.coroutine
	def delete(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		if os.path.isdir(real_path):
			shutil.rmtree(real_path)
		else:
			os.remove(real_path)

	@tornado.gen.coroutine
	def get(self, path):
		def force_download(buff, filename):
			self.set_header('Content-Type', 'application/force-download')
			self.set_header('Content-Disposition', 'attachment; filename=%s' % filename)
			self.finish(buff)

		real_path = self.manager.sd_card.resolve_path(path)

		if os.path.isfile(real_path):
			with open(real_path, "rb") as f:
				force_download(f.read(), os.path.basename(real_path))
		else:
			raise tornado.web.HTTPError(404, "File does not exist")


class MachineBedMeshHeightMapHandler(RestHandler):
	@tornado.gen.coroutine
	def get(self):
		bed_mesh = self.manager.printer.lookup_object("bed_mesh")
		if bed_mesh and bed_mesh.z_mesh and bed_mesh.z_mesh.mesh_z_table:
			height_map = yield self.get_height_map(bed_mesh)
			self.finish(height_map)
		else:
			raise tornado.web.HTTPError(404, "No height map available")

	@tornado.gen.coroutine
	def get_height_map(self, bed_mesh):
		z_mesh = bed_mesh.z_mesh

		mesh = []
		for y in range(z_mesh.mesh_y_count - 1, -1, -1):
			for x, z in enumerate(z_mesh.mesh_z_table[y]):
				mesh.append(
					[z_mesh.mesh_x_min + x * z_mesh.mesh_x_dist, z_mesh.mesh_y_min + (y - 1) * z_mesh.mesh_y_dist, (z)])


		probed = []
		for y, line in enumerate(bed_mesh.calibrate.probed_z_table):
			for x, z in enumerate(line):
				probed.append([z_mesh.mesh_x_min + x * z_mesh.mesh_x_dist, z_mesh.mesh_y_min + (y - 1) * z_mesh.mesh_y_dist, z ])

		return dict(mesh=mesh, probed=probed)


class MachineFileInfoHandler(RestHandler):
	@tornado.gen.coroutine
	def get(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		meta = self.manager.sd_card.parse_gcode_file(real_path)
		self.finish(meta)


class MachineCodeHandler(RestHandler):
	@tornado.gen.coroutine
	def post(self):
		responses = self.manager.dispatch(self.manager.process_gcode, self.request.body)
		self.set_header("Content-Type", "text/plain")
		self.finish("\n".join(responses))

class DummyHandler(RestHandler):
	@tornado.gen.coroutine
	def get(self):
		pass

# Legacy endpoints

class RRGCodeHandler(RestHandler):
	@tornado.gen.coroutine
	def get(self):
		# just replace M32 with PRINT_FILE, this endpoint is mostly used for that
		gcode = re.sub(r'M32\s+(\"[^"]+\")', r'PRINT_FILE FILE=\1', self.get_argument("gcode"))
		responses = self.manager.dispatch(self.manager.process_gcode, gcode)
		self.set_header("Content-Type", "text/plain")
		self.finish("\n".join(responses))

class RRUploadHandler(RestHandler):
	@tornado.gen.coroutine
	def post(self):
		path = self.get_argument("name")
		real_path = self.manager.sd_card.resolve_path(path)
		with open(real_path, 'w') as file:
			file.write(self.request.body)

class WebSocketHandler(tornado.websocket.WebSocketHandler):
	clients = set()

	def initialize(self, manager):
		self.manager = manager

	def check_origin(self, origin):
		return True

	def open(self):
		WebSocketHandler.clients.add(self)

	def on_message(self, message):
		if message == "PING\n":
			try:
				self.write_message("PONG\n")
			except:
				logging.exception("unable to write")
		elif message == "OK\n":
			# client ack, re-add it for broadcasting
			WebSocketHandler.clients.add(self)

	def on_close(self):
		try:
			WebSocketHandler.clients.remove(self)
		except:
			pass

	@classmethod
	def broadcast(cls, payload):
		for client in copy.copy(cls.clients):
			try:
				# remove the client and wait for reply before sending updates again
				WebSocketHandler.clients.remove(client)
				client.write_message(payload)
			except:
				logging.exception("unable to write")


class Job:
	def __init__(self, sd_card):
		self.sd_card = sd_card
		self.printer = sd_card.manager.printer
		self.printer.register_event_handler("klippy:shutdown", self.handle_shutdown)

		self.gcode = self.printer.lookup_object('gcode')

		self.gcode.register_command('PRINT_FILE', self.cmd_PRINT_FILE)
		self.gcode.register_command('SELECT_FILE', self.cmd_SELECT_FILE)

		# de-register any previously registered PAUSE/RESUME commands by pause_resume module
		self.gcode.register_command("PAUSE", None)
		self.gcode.register_command("RESUME", None)

		self.gcode.register_command("PAUSE", self.cmd_PAUSE)
		self.gcode.register_command("RESUME", self.cmd_RESUME)
		self.gcode.register_command("ABORT", self.cmd_ABORT)

		self.reactor = self.printer.get_reactor()

		self.fd = None
		self.last_file_name = None
		self.last_file_aborted = False
		self.last_file_cancelled = False

		self.reset()

	def handle_shutdown(self):
		logging.info("printer shutdown, aborting job")
		if self.is_processing():
			self.pause()
		else:
			self.handle_end()

	def reset(self):
		if self.fd is not None:
			self.fd.close()
		self.fd = None
		self.file_position = 0
		self.file = None
		self.did_abort = False
		self.did_pause = True
		self.work_timer = None

	def ensure_idle(self):
		if self.is_processing():
			raise self.gcode.error("SD busy")

	def select_file(self, filename):
		self.ensure_idle()
		self.reset()
		try:
			self.file = self.sd_card.parse_gcode_file(filename)
			self.fd = open(filename, 'rb')
			self.file_position = 0
		except:
			raise self.gcode.error("Failed to open file")

	def is_processing(self):
		return self.work_timer is not None

	def did_select_file(self):
		return self.fd and self.file

	def pause(self):
		if self.is_processing():
			self.did_pause = True

	def resume(self):
		if self.did_select_file:
			self.did_pause = False
			if not self.is_processing():
				self.work_timer = self.reactor.register_timer(self.work_handler, self.reactor.NOW)

	def abort(self):
		if self.did_select_file():
			if self.is_processing():
				self.did_abort = True
			else:
				self.handle_end(True)

	def get_status(self):
		if self.did_select_file():
			if self.is_processing():
				return 'pausing' if self.did_pause else 'processing'
			else:
				return 'paused' if self.did_pause else 'resuming'
		return 'idle'

	def get_state(self, eventtime):
		file_progress = 0.

		if self.did_select_file():
			file_progress = self.file_position / self.file["size"]

		return {
			"file": self.file,
			"filePosition": self.file_position,
			"warmUpDuration": 0,  # TODO
			"lastFileName": self.last_file_name,
			"lastFileAborted": self.last_file_aborted,
			"lastFileCancelled": self.last_file_cancelled,
			# "extrudedRaw": [],
			# "layer": null,
			# "layerTime": null,
			# "layers": [],
			"timesLeft": {
				"file": file_progress,
				"filament": 0,  # TODO
				"layer": 0,  # TODO
			}
		}

	def handle_end(self, did_abort=False):
		self.last_file_name = self.file["fileName"]
		self.last_file_aborted = did_abort
		self.last_file_cancelled = did_abort
		self.reset()
		if did_abort:
			template = self.sd_card.manager.abort_gcode
			if template:
				self.gcode.run_script_from_command(template.render())

	def work_handler(self, eventtime):
		logging.info("work handler started at position %d", self.file_position)

		self.reactor.unregister_timer(self.work_timer)

		try:
			self.fd.seek(self.file_position)
		except:
			logging.exception("work handler seek error")
			self.gcode.respond_error("Unable to seek file")
			self.work_timer = None
			return self.reactor.NEVER

		partial_input = ""
		lines = []
		while not (self.did_pause or self.did_abort):
			if not lines:
				try:
					data = self.fd.read(8192)
				except:
					logging.exception("work handler read error")
					self.gcode.respond_error("Unable to read file")
					break

				if not data:
					self.handle_end()
					break

				lines = data.split('\n')
				lines[0] = partial_input + lines[0]
				partial_input = lines.pop()
				lines.reverse()
				self.reactor.pause(self.reactor.NOW)
				continue

			# Pause if any other request is pending in the gcode class
			if self.gcode.get_mutex().test():
				logging.info("work handler paused, waiting for gcode mutex")
				self.reactor.pause(self.reactor.monotonic() + 0.100)
				continue

			# Dispatch command
			try:
				self.gcode.run_script(lines[-1])
			except self.gcode.error as e:
				break
			except:
				logging.exception("work handler dispatch error")
				break

			self.file_position += len(lines.pop()) + 1

		if self.did_abort:
			self.handle_end(True)

		logging.info("work handler ended at position %d", self.file_position)

		self.work_timer = None
		return self.reactor.NEVER

	def cmd_PRINT_FILE(self, params):
		try:
			self.select_file(self.sd_card.resolve_path(params["FILE"]))
			self.resume()
			self.gcode.respond("Print started: %s Size: %d" % (self.file["fileName"], self.file["size"]))
		except:
			raise self.gcode.error("Unable to open file")

	def cmd_SELECT_FILE(self, params):
		try:
			self.select_file(self.sd_card.resolve_path(params["FILE"]))
			self.gcode.respond("File opened: %s Size: %d" % (self.file["fileName"], self.file["size"]))
		except:
			raise self.gcode.error("Unable to open file")


	def cmd_PAUSE(self, params):
		if self.did_select_file():
			if self.did_pause:
				self.gcode.respond_info("Print already paused")
			else:
				self.gcode.run_script_from_command("SAVE_GCODE_STATE STATE=PAUSE_STATE")
				self.pause()
		else:
			self.gcode.respond_error("No job in progress to be paused")


	def cmd_RESUME(self, params):
		if self.did_select_file():
			if self.did_pause:
				self.gcode.run_script_from_command("RESTORE_GCODE_STATE STATE=PAUSE_STATE MOVE=1")
				self.resume()
			else:
				self.gcode.respond_info("Print is not paused, resume ignored")
		else:
			self.gcode.respond_error("No job in progress to be resumed")


	def cmd_ABORT(self, params):
		if self.did_select_file():
			self.abort()
			self.gcode.respond_info("Print aborted")
		else:
			self.gcode.respond_error("No job in progress to be aborted")


class SDCard:
	def __init__(self, manager):
		self.manager = manager
		self.printer = self.manager.printer
		self.root_path = os.path.normpath(os.path.expanduser(manager.config.get('path')))
		self.job = Job(self)
		self.gcode = self.printer.lookup_object('gcode')
		self.gcode.register_command('RUN_MACRO', self.cmd_RUN_MACRO)

	def resolve_path(self, path):
		if not path.startswith("0:"):
			raise Exception("Invalid path")
		return re.sub(r"^0:", self.root_path, path)

	def virtual_path(self, path):
		return "0:/" + os.path.relpath(path, self.root_path)

	def parse_gcode_file(self, path):
		with open(path, 'rb') as f:
			return dict(
				parse_gcode(f.read()),
				fileName="0:/" + os.path.relpath(path, self.root_path),
				size=os.stat(path).st_size,
				lastModified=datetime.utcfromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%dT%H:%M:%S"),
			)

	def cmd_RUN_MACRO(self, params):
		real_path = self.resolve_path(params["FILE"])
		try:
			with open(real_path, "rb") as f:
				content = f.read()
		except:
			raise self.gcode.error("Unable to read file")

		try:
			env = jinja2.Environment('{%', '%}', '{', '}')
			template = TemplateWrapper(self.manager.printer, env, params["FILE"], content)
			macro = template.render()
		except Exception as e:
			raise self.gcode.error(str(e))

		self.gcode.run_script_from_command(macro)



class ToolState:
	def __init__(self, manager):
		self.manager = manager
		self.printer = manager.printer
		self.printer.register_event_handler("klippy:ready", self.handle_ready)

		self.extruders = []

	def handle_ready(self):
		self.extruders = kinematics.extruder.get_printer_extruders(self.printer)

	def get_state(self, eventtime):
		tools = []
		for extruder in self.extruders:
			ex_heater_index = self.manager.heat.get_heater_index(extruder.heater)
			heater_status  = extruder.heater.get_status(eventtime)

			tools.append({
				"number": self.extruders.index(extruder),
				"active": [heater_status['target']],
				# "name": extruder.get_name(),
				"filamentExtruder": 0,
				# "filament": null,
				"fans": [0],  # TODO: Figure out a way to lookup extruder fans
				"heaters": [ex_heater_index],
				"extruders": [self.get_extruder_index(extruder)],
				# "mix": [],
				# "axes": [],
				# "offsets": [], # TODO
				# "offsetsProbed": 0 # TODO
			})
		return tools

	def get_extruder_index(self, extruder):
		# TODO: handle missing extruder?
		return self.extruders.index(extruder)


class MoveState:
	def __init__(self, manager):
		self.manager = manager
		self.printer = manager.printer
		self.printer.register_event_handler("klippy:ready", self.handle_ready)

		self.gcode = self.printer.lookup_object("gcode")
		self.configfile = self.printer.lookup_object('configfile').read_main_config()

		self.extruders = []
		self.kinematics = None
		self.bed_mesh = None
		self.top_speed = 0

	def handle_ready(self):
		self.toolhead = self.printer.lookup_object('toolhead')
		self.kinematics = self.toolhead.get_kinematics()
		self.bed_mesh = self.printer.lookup_object("bed_mesh", None)
		self.extruders = kinematics.extruder.get_printer_extruders(self.printer)
		self.top_speed = 0

	def get_state(self, eventtime):
		position = self.gcode.last_position
		gcode_status = self.gcode.get_status(eventtime)

		requested_speed = gcode_status["speed"] / 60

		if requested_speed > self.top_speed:
			self.top_speed = requested_speed

		steppers = []
		drives = []
		axes = []
		if self.kinematics:
			for rail in self.kinematics.rails:
				min_pos, max_pos = rail.get_range()
				low_limit, high_limit = self.kinematics.limits[self.kinematics.rails.index(rail)]

				for stepper in rail.steppers:
					steppers.append(stepper)
					drives.append({
						"position": position[self.kinematics.rails.index(rail)],
						# "microstepping": { "value": 16, "interpolated": true },
						# "current": null,
						# "acceleration": null,
						# "minSpeed": null,
						# "maxSpeed": null
					})

				axes.append({
					"letter": rail.name,
					"drives": [steppers.index(stepper) for stepper in rail.steppers],
					"homed": low_limit <= high_limit,
					# "machinePosition": null,
					"min": min_pos,
					# "minEndstop": null,
					# "minProbed": false,
					"max": max_pos,
					# "maxEndstop": null,
					# "maxProbed": false,
					# "visible": true
				})


		extruders = []
		for extruder in self.extruders:
			steppers.append(extruder.stepper)
			drives.append({
				"position": position[steppers.index(extruder.stepper)],
				# "microstepping": { "value": 16, "interpolated": true },
				# "current": null,
				# "acceleration": null,
				# "minSpeed": null,
				# "maxSpeed": null
			})
			extruders.append({
				"drives": [steppers.index(extruder.stepper)],
				"factor": self.gcode.extrude_factor,
				# "nonlinear": { "a": 0, "b": 0, "upperLimit": 0.2, "temperature": 0 }
			})

		return ({
			"axes": axes,
			"drives": drives,
			"extruders": extruders,
			"babystepZ": self.gcode.homing_position[2],
			"currentMove": {"requestedSpeed": requested_speed, "topSpeed": self.top_speed},
			"compensation": not not (self.bed_mesh and self.bed_mesh.z_mesh),
			"speedFactor": gcode_status["speed_factor"],
			"geometry": {
				"type": self.configfile.getsection("printer").get("kinematics"),
			}
		})


class HeatState:
	def __init__(self, manager):
		self.printer = manager.printer
		self.printer.register_event_handler("klippy:ready", self.handle_ready)

		self.heat = self.printer.lookup_object('heater')
		self.heaters = []
		self.heat_beds = []
		self.probe_temps = []

	def handle_ready(self):
		self.heaters = self.heat.heaters.items()
		self.heat_beds = self.printer.lookup_objects('heater_bed')
		self.probe_temps = self.printer.lookup_objects('probe_temp')

	def get_state(self, eventtime):
		heater_statuses = [heater.get_status(eventtime) for name, heater in self.heaters]

		beds = []
		for name, heat_bed in self.heat_beds:
			bed_heater_index = self.get_heater_index(heat_bed)
			heater_bed_status = heater_statuses[bed_heater_index]
			beds.append({
				"active": [heater_bed_status['target']],
				# "name": name,
				"heaters": [bed_heater_index]
			})

		heaters = []
		if self.heat:
			for name, heater in self.heaters:
				heater_status = heater_statuses[self.get_heater_index(heater)]
				heaters.append({
					"current": heater_status["temperature"],
					"name": name,
					"state": self.get_heater_state(heater_status),
				})

		extra = []
		for name, probe_temp in self.probe_temps:
			probe_temp, probe_target = probe_temp.get_temp(eventtime)
			extra.append({
				"name": "probe_temp",
				"current": probe_temp
			})

		return ({
			"beds": beds,
			"heaters": heaters,
			"extra": extra,
		})

	def get_heater_index(self, heater):
		return self.heat.heaters.values().index(heater)

	def get_heater_state(self, status):
		state = 0
		if status['target'] > 0:
			state = 1
		return state


class SensorState:
	def __init__(self, manager):
		self.manager = manager
		self.printer = manager.printer
		self.printer.register_event_handler("klippy:ready", self.handle_ready)
		self.probes = []

	def handle_ready(self):
		query_endstops = self.printer.try_load_module(self.manager.config, 'query_endstops')
		self.endstops = query_endstops.endstops
		self.probes = self.printer.lookup_objects('probe')

	def get_state(self, eventtime):
		probes = []

		for name, probe in self.probes:
			probes.append({
				# "type": null,
				# "value": null,
				# "secondaryValues": [],
				# "threshold": 500,
				# "speed": 2,
				# "diveHeight": 5,
				"offsets": probe.get_offsets(),
				# "triggerHeight": 0.7,
				# "filtered": true,
				# "inverted": false,
				# "recoveryTime": 0,
				# "travelSpeed": 100,
				# "maxProbeCount": 1,
				# "tolerance": 0.03,
				# "disablesBed": false,
				# "persistent": false
			})

		endstops = []
		# last_move_time = self.toolhead.get_last_move_time() # TODO: raises random exceptions like DripModeEndSignal
		# for endstop, name in self.endstops:
		# 	endstops.append({"name": name, "triggered": endstop.query_endstop(last_move_time)})

		return ({
			"endstops": endstops,
			"probes": probes
		})


class FanState:
	def __init__(self, manager):
		self.manager = manager
		self.printer = self.manager.printer
		self.printer.register_event_handler("klippy:ready", self.handle_ready)
		self.fans = []
		self.heater_fans = []

	def handle_ready(self):
		self.fans = self.printer.lookup_objects('fan')
		self.heater_fans = self.printer.lookup_objects('heater_fan')

	def get_state(self, eventtime):
		fans = []

		for name, fan in self.fans:
			fan_status = fan.get_status(eventtime)
			fans.append({
				"name": name,
				"value": fan_status["speed"],
				"max": fan.max_power
				# 	rpm = null
				# 	inverted = false
				# 	frequency = null
				# 	min = 0.0
				# 	max = 1.0
				# 	blip = 0.1
				# 	pin = null
			})

		for name, fan in self.heater_fans:
			fan_status = fan.get_status(eventtime)
			fans.append({
				"name": name,
				"value": fan_status["speed"],
				"max": fan.fan.max_power,
				"thermostatic": {
					"control": True,
					"heaters": [self.manager.heat.get_heater_index(heater) for heater in fan.heaters],
					"temperature": fan.heater_temp
				}
			})

		return fans


class State:
	def __init__(self, manager):
		self.manager = manager
		self.printer = manager.printer
		self.sd_card = manager.sd_card
		self.printer.register_event_handler("klippy:ready", self.handle_ready)
		self.status = 'off'

		self.toolhead = None
		self.gcode = self.printer.lookup_object("gcode")

	def handle_ready(self):
		self.status = 'idle'
		self.toolhead = self.printer.lookup_object('toolhead', None)

	def handle_disconnect(self):
		self.status = 'off'

	def get_state(self, eventtime):
		return ({
			"status": self.get_status(),
			"currentTool": self.manager.tools.get_extruder_index(self.toolhead.extruder) if self.toolhead else None,
			# "displayMessage": null, # TODO: used?
			"logFile": None,  # TODO: figure out how to get it
			# "mode": "FFF",  # TODO: useful?
		})

	def get_status(self):
		# 'updating';
		# 'changingTool'
		status = self.status

		if self.printer.is_shutdown:
			return 'off'

		if self.gcode.is_processing_data:
			status = 'busy'

		if self.sd_card.job.did_select_file():
			status = self.sd_card.job.get_status()

		return status


class Manager:
	def __init__(self, config):
		self.config = config

		self.printer = config.get_printer()

		self.printer_name = config.get('printer_name', "Klipper")

		self.gcode_macro = self.printer.try_load_module(config, 'gcode_macro')

		if config.get('abort_gcode', None) is not None:
			self.abort_gcode = self.gcode_macro.load_template(config, 'abort_gcode')

		# filament_switch_sensor depends on pause_resume, we load it first and re-attach to our own implementation
		self.printer.try_load_module(config, 'pause_resume')

		self.sd_card = SDCard(self)

		self.reactor = self.printer.get_reactor()
		self.timer = self.reactor.register_timer(self.handle_timer)

		self.gcode = self.printer.lookup_object('gcode')
		self.printer.register_event_handler("gcode:response", self.handle_gcode_response)

		self.state = State(self)
		self.tools = ToolState(self)
		self.move = MoveState(self)
		self.heat = HeatState(self)
		self.fans = FanState(self)
		self.sensors = SensorState(self)

		self.gcode_responses = []

		# blocks concurrent execution of multiple web gcode invocations
		self.process_mutex = self.reactor.mutex()

		self.broadcast_queue = Queue()
		self.broadcast_thread = threading.Thread(target=self.broadcast_loop)
		self.broadcast_thread.start()

		self.reactor.update_timer(self.timer, self.reactor.NOW)

	def handle_timer(self, eventtime):
		self.broadcast_queue.put_nowait(self.get_state(eventtime))
		return eventtime + .25

	def broadcast_loop(self):
		while True:
			state = self.broadcast_queue.get(True)
			if len(WebSocketHandler.clients) > 0:
				WebSocketHandler.broadcast(state)

	def handle_gcode_response(self, msg):
		if re.match('(B|T\d):\d+.\d\s/\d+.\d+', msg): return
		self.gcode_responses.append(msg)

	# used to run commands within the reactor from different threads
	def dispatch(self, target, *args):
		q = JoinableQueue()

		def callback(e):
			q.put(target(*args))
			q.task_done()

		reactor = self.printer.get_reactor()
		reactor.register_async_callback(callback)

		q.join()
		return q.get()

	def process_gcode(self, gcode):
		responses = []

		with self.process_mutex:
			try:
				previous_responses = self.gcode_responses
				self.gcode_responses = []

				with self.gcode.get_mutex():
					self.gcode._process_commands(gcode.split('\n'))

			finally:
				responses = self.gcode_responses
				self.gcode_responses = previous_responses

		return responses

	def get_messages(self):
		messages = copy.copy(self.gcode_responses)
		del self.gcode_responses[:]
		return messages

	def get_state(self, eventtime):
		return ({
			"messages": self.get_messages(),
			"state": self.state.get_state(eventtime),
			"tools": self.tools.get_state(eventtime),
			"fans": self.fans.get_state(eventtime),
			"heat": self.heat.get_state(eventtime),
			"move": self.move.get_state(eventtime),
			"job": self.sd_card.job.get_state(eventtime),
			"sensors": self.sensors.get_state(eventtime),
			"network": {
				"name": self.printer_name,
				# "hostname": "klipper",
				# "password": "reprap",
				# "interfaces": []
			},
			"storages": [{
				"mounted": True
			}],
		})


class KlipperWebControl:
	def __init__(self, config):
		self.config = config
		self.printer = self.config.get_printer()

		self.address = self.config.get('address', "127.0.0.1")
		self.port = self.config.getint("port", 4444)
		self.manager = Manager(self.config)
		self.app =	tornado.web.Application([
			# legacy endpoints just third party integration
			("/rr_connect", DummyHandler, {"manager": self.manager}),
			("/rr_disconnect", DummyHandler, {"manager": self.manager}),
			("/rr_upload", RRUploadHandler, {"manager": self.manager}),
			("/rr_gcode", RRGCodeHandler, {"manager": self.manager}),

			("/machine/bed_mesh/height_map", MachineBedMeshHeightMapHandler, {"manager": self.manager}),
			("/machine/file/move", MachineMoveHandler, {"manager": self.manager}),
			(r"/machine/file/(.*)", MachineFileHandler, {"manager": self.manager}),
			(r"/machine/fileinfo/(.*)", MachineFileInfoHandler, {"manager": self.manager}),
			(r"/machine/directory/(.*)", MachineDirectoryHandler, {"manager": self.manager}),
			("/machine/code", MachineCodeHandler, {"manager": self.manager}),
			("/machine", WebSocketHandler, {"manager": self.manager}),
			(r"/.*", WebRootRequestHandler, {"manager": self.manager}),
		], cookie_secret=base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes))

		self.thread = None
		self.ioloop = None
		self.http_server = None

		self.handle_ready()
		self.printer.register_event_handler("klippy:ready", self.handle_ready)
		self.printer.register_event_handler("klippy:disconnect", self.handle_disconnect)

	def handle_ready(self):
		if not self.thread or not self.thread.is_alive:
			self.thread = threading.Thread(target=self.spawn)
			self.thread.start()

	def handle_disconnect(self):
		if self.ioloop:
			self.ioloop.stop()

		if self.http_server:
			self.http_server.stop()

	def spawn(self):
		logging.info("KWC starting at http://%s:%s", self.address, self.port)
		self.http_server = tornado.httpserver.HTTPServer(self.app, max_buffer_size=500 * 1024 * 1024)
		self.http_server.listen(self.port)
		self.ioloop = tornado.ioloop.IOLoop.current()
		self.ioloop.start()
		logging.info("KWC stopped.")


def load_config(config):
	return KlipperWebControl(config)
