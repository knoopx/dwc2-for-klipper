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

from gcode import GCodeParser

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

	def options(self, *kwargs):
		self.set_status(204)
		self.finish()


class MachineMoveHandler(RestHandler):
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

	def put(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		if not os.path.exists(real_path):
			os.makedirs(real_path)


class MachineFileHandler(RestHandler):
	def put(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		with open(real_path, 'w') as file:
			file.write(self.request.body)

	def delete(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		if os.path.isdir(real_path):
			shutil.rmtree(real_path)
		else:
			os.remove(real_path)

	def get(self, path):
		def force_download(buff, filename):
			self.set_header('Content-Type', 'application/force-download')
			self.set_header('Content-Disposition', 'attachment; filename=%s' % filename)
			self.finish(buff)

		real_path = self.manager.sd_card.resolve_path(path)

		bed_mesh = self.manager.printer.lookup_object("bed_mesh")

		if bed_mesh and path == '0:/sys/heightmap.csv':
			force_download(self.get_height_map_csv(bed_mesh), "heightmap.csv")
		elif os.path.isfile(real_path):
			with open(real_path, "rb") as f:
				force_download(f.read(), os.path.basename(real_path))
		else:
			raise tornado.web.HTTPError(404, "File does not exist")

	def get_height_map_csv(self, bed_mesh):
		def calc_mean(matrix_):
			matrix_tolist = []
			for line in matrix_:
				matrix_tolist += line
			return float(sum(matrix_tolist)) / len(matrix_tolist)

		def calc_stdv(matrix_):
			from math import sqrt
			matrix_tolist = []
			for line in matrix_:
				matrix_tolist += line

			mean = float(sum(matrix_tolist)) / len(matrix_tolist)
			return sqrt(float(reduce(lambda x, y: x + y, map(lambda x: (x - mean) ** 2, matrix_tolist))) / len(
				matrix_tolist))  # Stackoverflow - liked that native short solution

		hmap = []
		z_matrix = bed_mesh.calibrate.probed_z_table
		mesh_data = bed_mesh.z_mesh

		meane_ = round(calc_mean(z_matrix), 3)
		stdev_ = round(calc_stdv(z_matrix), 3)

		hmap.append('RepRapFirmware height map file v2 generated at ' + str(
			datetime.now().strftime('%Y-%m-%d %H:%M')) + ', mean error ' + str(meane_) + ', deviation ' + str(
			stdev_))
		hmap.append('xmin,xmax,ymin,ymax,radius,xspacing,yspacing,xnum,ynum')
		xspace_ = (mesh_data.mesh_x_max - mesh_data.mesh_x_min) / mesh_data.mesh_x_count
		yspace_ = (mesh_data.mesh_y_max - mesh_data.mesh_y_min) / mesh_data.mesh_y_count
		hmap.append(str(mesh_data.mesh_x_min) + ',' + str(mesh_data.mesh_x_max) + ',' + str(
			mesh_data.mesh_y_min) + ',' + str(mesh_data.mesh_y_max) + \
		            ',-1.00,' + str(xspace_) + ',' + str(yspace_) + ',' + str(mesh_data.mesh_x_count) + ',' + str(
			mesh_data.mesh_y_count))

		for line in z_matrix:
			red_by_offset = map(lambda x: x - meane_, line)
			hmap.append('  ' + ',  '.join(map(str, red_by_offset)))

		return "\n".join(hmap)


class MachineFileInfoHandler(RestHandler):
	def get(self, path):
		real_path = self.manager.sd_card.resolve_path(path)
		meta = self.manager.sd_card.parse_gcode_file(real_path)
		self.finish(meta)


class MachineCodeHandler(RestHandler):
	@tornado.web.asynchronous
	def post(self):
		def handler(e):
			responses = self.manager.dispatch_gcode(self.request.body)
			self.set_header("Content-Type", "text/plain")
			self.finish("\n".join(responses))

		reactor = self.manager.printer.get_reactor()
		reactor.register_async_callback(handler)

class DummyHandler(RestHandler):
	def get(self):
		pass

# Legacy endpoints

class RRGCodeHandler(RestHandler):
	@tornado.web.asynchronous
	def get(self):
		# just replace M32 with PRINT_FILE, this endpoint is mostly used for that
		gcode = re.sub(r'M32\s+(\"[^"]+\")', r'PRINT_FILE FILE=\1', self.get_argument("gcode"))

		def handler(e):
			responses = self.manager.dispatch_gcode(gcode)
			self.set_header("Content-Type", "text/plain")
			self.finish("\n".join(responses))

		reactor = self.manager.printer.get_reactor()
		reactor.register_async_callback(handler)

class RRUploadHandler(RestHandler):
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
		WebSocketHandler.clients.remove(self)

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
		printer = sd_card.manager.printer
		printer.register_event_handler("klippy:shutdown", self.handle_shutdown)

		self.toolhead = printer.lookup_object('gcode')
		self.gcode = printer.lookup_object('gcode')

		self.gcode.register_command('PRINT_FILE', self.cmd_PRINT_FILE)
		self.gcode.register_command('SELECT_FILE', self.cmd_SELECT_FILE)

		# de-register any previously registered PAUSE/RESUME commands by pause_resume module
		self.gcode.register_command("PAUSE", None)
		self.gcode.register_command("RESUME", None)

		self.gcode.register_command("PAUSE", self.cmd_PAUSE)
		self.gcode.register_command("RESUME", self.cmd_RESUME)
		self.gcode.register_command("ABORT", self.cmd_ABORT)

		self.reactor = printer.get_reactor()

		self.fd = None
		self.last_file_name = None
		self.last_file_aborted = False
		self.last_file_cancelled = False

		self.reset()

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
		self.did_pause = False
		if not self.is_processing():
			self.work_timer = self.reactor.register_timer(self.work_handler, self.reactor.NOW)

	def abort(self):
		self.did_abort = True
		self.resume()

	def handle_shutdown(self):
		self.abort()

	def get_status(self):
		if self.is_processing():
			return 'pausing' if self.did_pause else 'processing'
		else:
			return 'paused' if self.did_pause else 'resuming'

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

	def work_handler(self, eventtime):
		logging.info("Resuming job at position %d", self.file_position)

		self.reactor.unregister_timer(self.work_timer)
		try:
			self.fd.seek(self.file_position)
		except:
			logging.exception("seek error")
			self.gcode.respond_error("Unable to seek file")
			self.work_timer = None
			return self.reactor.NEVER
		gcode_mutex = self.gcode.get_mutex()
		partial_input = ""
		lines = []
		while not (self.did_pause or self.did_abort):
			if not lines:
				# Read more data
				try:
					data = self.fd.read(8192)
				except:
					logging.exception("job read error")
					self.gcode.respond_error("Unable to read file")
					break
				if not data:
					# End of file
					self.last_file_name = self.file["fileName"]
					self.last_file_aborted = False
					self.last_file_cancelled = False
					self.reset()
					logging.info("job finished")
					self.gcode.respond("Job finished")

					break
				lines = data.split('\n')
				lines[0] = partial_input + lines[0]
				partial_input = lines.pop()
				lines.reverse()
				self.reactor.pause(self.reactor.NOW)
				continue

			# Pause if any other request is pending in the gcode class
			if gcode_mutex.test():
				self.reactor.pause(self.reactor.monotonic() + 0.100)
				continue
			# Dispatch command
			try:
				self.gcode.run_script(lines[-1])
			except self.gcode.error as e:
				break
			except:
				logging.exception("job dispatch error")
				break
			self.file_position += len(lines.pop()) + 1

		if self.did_abort:
			logging.info("job aborted at position %d", self.file_position)
			self.last_file_name = self.file["fileName"]
			self.last_file_aborted = True
			self.last_file_cancelled = True
			self.reset()

			template = self.sd_card.manager.abort_gcode
			if template:
				self.gcode.run_script(template.render())
		else:
			logging.info("job paused at position %d", self.file_position)

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
		if self.did_pause:
			self.gcode.respond_info("Print already paused")
		elif self.did_select_file():
			self.gcode.run_script_from_command("SAVE_GCODE_STATE STATE=PAUSE_STATE")
			self.pause()
		else:
			self.gcode.respond_error("No job in progress to be paused")


	def cmd_RESUME(self, params):
		if not self.did_pause:
			self.gcode.respond_info("Print is not paused, resume ignored")
		elif self.did_select_file():
			self.gcode.run_script_from_command("RESTORE_GCODE_STATE STATE=PAUSE_STATE MOVE=1")
			self.resume()
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
		self.job = Job(self)
		self.root_path = os.path.normpath(os.path.expanduser(manager.config.get('path')))
		self.gcode = manager.printer.lookup_object('gcode')
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

		self.configfile = self.printer.lookup_object('configfile').read_main_config()
		self.toolhead = self.printer.lookup_object('toolhead')
		self.kinematics = self.toolhead.get_kinematics()
		self.extruders = kinematics.extruder.get_printer_extruders(self.printer)

	def get_state(self, eventtime):
		tools = []
		for extruder in self.extruders:
			ex_heater_index = self.manager.heat.get_heater_index(extruder.heater)
			tools.append({
				"number": self.extruders.index(extruder),
				"active": [0],
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

		self.configfile = self.printer.lookup_object('configfile').read_main_config()
		self.gcode = self.printer.lookup_object("gcode")
		self.toolhead = self.printer.lookup_object('toolhead')
		self.kinematics = self.toolhead.get_kinematics()
		self.extruders = kinematics.extruder.get_printer_extruders(self.printer)
		self.bed_mesh = self.printer.lookup_object("bed_mesh", None)

		self.top_speed = 0

	def get_state(self, eventtime):
		position = self.gcode.last_position
		gcode_status = self.gcode.get_status(eventtime)

		requested_speed = gcode_status["speed"] / 60

		if requested_speed > self.top_speed:
			self.top_speed = requested_speed

		steppers = []
		drives = []

		for rail in self.kinematics.rails:
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

		axes = []
		for rail in self.kinematics.rails:
			min_pos, max_pos = rail.get_range()
			low_limit, high_limit = self.kinematics.limits[self.kinematics.rails.index(rail)]

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
		self.heat = self.printer.lookup_object('heater')
		self.heaters = self.heat.heaters.items()
		self.heat_beds = self.printer.lookup_objects('heater_bed')
		self.probe_temp = self.printer.lookup_object('probe_temp', None)

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
		for name, heater in self.heat.heaters.items():
			heater_status = heater_statuses[self.get_heater_index(heater)]
			heaters.append({
				"current": heater_status["temperature"],
				"name": name,
				"state": self.get_heater_state(heater_status),
			})

		extra = []
		if self.probe_temp:
			probe_temp, probe_target = self.probe_temp.get_temp(eventtime)
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
		manager = 0
		if status['target'] > 0:
			manager = 1
		return manager


class SensorState:
	def __init__(self, manager):
		printer = manager.printer
		query_endstops = printer.try_load_module(manager.config, 'query_endstops')
		self.endstops = query_endstops.endstops
		self.probe = printer.lookup_object('probe', None)

	def get_state(self, eventtime):
		probes = []

		if self.probe:
			probes.append({
				# "type": null,
				# "value": null,
				# "secondaryValues": [],
				# "threshold": 500,
				# "speed": 2,
				# "diveHeight": 5,
				"offsets": [self.probe.z_offset],
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

	def get_state(self, eventtime):
		fans = []

		# 	rpm = null
		# 	inverted = false
		# 	frequency = null
		# 	min = 0.0
		# 	max = 1.0
		# 	blip = 0.1
		# 	pin = null

		for name, fan in self.printer.lookup_objects('fan'):
			fan_status = fan.get_status(eventtime)
			fans.append({
				"name": name,
				"value": fan_status["speed"],
				"max": fan.max_power
			})

		for name, fan in self.printer.lookup_objects('heater_fan'):
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
		self.gcode = manager.printer.lookup_object("gcode")
		self.toolhead = manager.printer.lookup_object('toolhead')

	def get_state(self, eventtime):
		return ({
			"currentTool": self.manager.tools.get_extruder_index(self.toolhead.extruder),
			# "displayMessage": null, # TODO: used?
			"logFile": None,  # TODO: figure out how to get it
			# "mode": "FFF",  # TODO: useful?
			"status": self.get_status(),
		})

	def get_status(self):
		# 'updating';
		# 'halted';
		# 'changingTool'
		status = 'idle'
		if 'Printer is ready' != self.printer.get_state_message():
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
		self.printer.register_event_handler("klippy:ready", self.handle_ready)

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

		self.state = None
		self.tools = None
		self.move = None
		self.heat = None
		self.fans = None
		self.sensors = None

		self.gcode_responses = []

	def handle_ready(self):
		self.state = State(self)
		self.tools = ToolState(self)
		self.move = MoveState(self)
		self.heat = HeatState(self)
		self.fans = FanState(self)
		self.sensors = SensorState(self)
		self.reactor.update_timer(self.timer, self.reactor.NOW)

	def handle_timer(self, eventtime):
		if len(WebSocketHandler.clients) > 0:
			WebSocketHandler.broadcast(self.get_state(eventtime))
		return eventtime + .25

	def handle_gcode_response(self, msg):
		self.gcode_responses.append(msg)

	def dispatch_gcode(self, gcode):
		responses = []
		with self.gcode.mutex:
			previous_responses = self.gcode_responses
			self.gcode_responses = []
			try:
				self.gcode._process_commands(gcode.split('\n'), need_ack=True)
			finally:
				responses = self.gcode_responses
				self.gcode_responses = previous_responses
		return responses

	def get_state(self, eventtime):
		return ({
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
		self.address = config.get('address', "127.0.0.1")
		self.port = config.getint("port", 4444)

		self.thread = None
		self.ioloop = None
		self.manager = Manager(self.config)

		printer = config.get_printer()
		printer.register_event_handler("klippy:ready", self.handle_ready)
		printer.register_event_handler("klippy:disconnect", self.handle_disconnect)

	def handle_disconnect(self):
		tornado.ioloop.IOLoop.current().stop()
		self.http_server.stop()

	def handle_ready(self):
		def spawn(address, port, app):
			logging.info("KWC starting at http://%s:%s", address, port)
			self.http_server = tornado.httpserver.HTTPServer(app, max_buffer_size=500 * 1024 * 1024)
			self.http_server.listen(self.port)
			tornado.ioloop.IOLoop.current().start()

		app = tornado.web.Application([
			# legacy endpoints just third party integration
			("/rr_connect", DummyHandler, {"manager": self.manager}),
			("/rr_disconnect", DummyHandler, {"manager": self.manager}),
			("/rr_upload", RRUploadHandler, {"manager": self.manager}),
			("/rr_gcode", RRGCodeHandler, {"manager": self.manager}),

			("/machine/file/move", MachineMoveHandler, {"manager": self.manager}),
			(r"/machine/file/(.*)", MachineFileHandler, {"manager": self.manager}),
			(r"/machine/fileinfo/(.*)", MachineFileInfoHandler, {"manager": self.manager}),
			(r"/machine/directory/(.*)", MachineDirectoryHandler, {"manager": self.manager}),
			("/machine/code", MachineCodeHandler, {"manager": self.manager}),
			("/machine", WebSocketHandler, {"manager": self.manager}),
			(r"/.*", WebRootRequestHandler, {"manager": self.manager}),
		], cookie_secret=base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes))

		self.thread = threading.Thread(target=spawn, args=(self.address, self.port, app))
		self.thread.daemon = True
		self.thread.start()


def load_config(config):
	return KlipperWebControl(config)
