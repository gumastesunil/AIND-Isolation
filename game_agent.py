"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    import numpy as np               # additional python modules
    from scipy.special import expit  # additional python modules
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    # W1 is the weight matrix occuring between the input layer and hidden layer. Dimension is 15x50
    # W1 is learnt using Temporal Difference learning after playing the agent against itself 100,000 times 
    W1 = np.asarray([[1.956246228952736310e-02,2.371917707980669004e-02,2.195739076971876813e-02,2.624433119140364562e-02,2.163926647441404699e-02,9.292599431565318549e-03,1.235144932612089637e-02,1.119859237639575233e-01,1.029715390618104603e-01,1.364857728067753878e-02,1.615361329919464961e-01,4.328840706466083321e-02,4.092019274656744782e-02,3.472171294857334667e-02,5.283959470937818481e-02,1.574997236550435276e-01,1.620267509579919052e-01,1.119761971091818298e-01,3.127909567034539706e-02,5.999749796399451995e-02,2.812815673576075925e-02,1.041470307119851113e-01,9.853280298655825442e-02,7.254272847655435230e-02,6.030712088545528426e-02,1.323404880580088261e-01,4.960163988433807791e-02,2.069832057695071906e-02,9.446586125091382957e-02,2.187864380627648625e-02,1.039813674245258363e-01,1.113323418073491611e-01,3.063085510916833260e-02,3.211207399457335909e-02,5.471072958649836337e-02,5.783858698305432977e-02,5.997123442501608309e-02,4.052810774242338737e-02,4.263572230512742423e-02,7.207214311907096660e-02,7.754682253292875804e-02,1.910504184048231191e-02,2.105838361494039024e-02,7.796860484598622243e-02,2.823393930322955561e-02,3.269130067226679037e-02,4.019984797541776428e-02,4.642959853328679221e-02,1.397093194779839252e-02,1.446927432569264094e-01],
[7.942034481455128095e-03,2.428210003804404452e-02,2.751107825317140529e-02,7.674223487744273771e-03,-2.914999257947681268e-04,1.382509223529040931e-02,2.236258752864338228e-02,1.156274164806029986e-01,1.174386557348199578e-01,1.163100661069985629e-02,1.779450514284553753e-01,1.032623957160147370e-02,1.269361291511466802e-02,2.621102616376105865e-02,4.020180899305787064e-02,1.754096672559489756e-01,1.475751740226241560e-01,1.218825479443980120e-01,3.209221417468723081e-02,7.309814097028041746e-02,3.947323888695495643e-03,1.118379109991094605e-01,9.338193776985460881e-02,6.155500998964957837e-02,5.341211995986303723e-02,1.416898379431404675e-01,4.053002262467265276e-02,1.782953765555775338e-02,1.102917987201311734e-01,5.177673336559630857e-02,1.042127676572838363e-01,8.742205993682464205e-02,5.991569550084429457e-02,5.134068382457326885e-02,6.120197948063708926e-02,7.090214906987042653e-02,5.582193405710064726e-02,1.533881358962321459e-02,4.042247470675636861e-02,5.447991536141999491e-02,9.312984419941673642e-02,2.007186235020673795e-02,3.628602096950440947e-02,7.756422322303752059e-02,3.417541284666719692e-02,2.706552180212036529e-02,3.560886131328256515e-02,5.051424120579053184e-02,1.691444502551052789e-02,1.537407171801581929e-01],
[2.538789288020033841e-02,2.984023811225045703e-02,3.848420491455029763e-02,2.735818092044647876e-02,1.513881945833961407e-02,1.566629430816503287e-03,3.715023501011123497e-02,1.295436668193456764e-01,1.125041288134052547e-01,2.344807442496899680e-02,1.674704210020482587e-01,3.787809226771079313e-02,4.298285776623306936e-02,3.805276571616874465e-02,4.400633837149994787e-02,1.504671005018050622e-01,1.572112453320128056e-01,1.127841570689641498e-01,1.067639933656470280e-02,5.456842876798511516e-02,1.769282414977614917e-02,1.249672754198264957e-01,7.859097218464161649e-02,7.903105167536685072e-02,5.359585920488512539e-02,1.406495267998389975e-01,3.738409225870133884e-02,4.152191636673203956e-02,9.840271198490400939e-02,6.351196360788170792e-02,1.057030884094207063e-01,7.628812616327348972e-02,5.004789267168199829e-02,5.506488869160883898e-02,5.218557597177824608e-02,5.738635628116710991e-02,6.334820455704470721e-02,4.132065198707374210e-02,1.987680740295228593e-02,5.637932548036261177e-02,1.010691949230727998e-01,1.444795140890494864e-02,3.591811575675961377e-02,7.708096102633432700e-02,2.009034232711493645e-02,4.561533551921045926e-02,1.545297977641764929e-02,4.999523691452673058e-02,1.275519586686820611e-02,1.450791647881752577e-01],
[1.949675863830969968e-02,3.942386432666859092e-02,2.390526972824219554e-02,2.569051969477019123e-02,3.693489253994376026e-02,2.675723716764317681e-02,3.950993068608176645e-02,1.328721282299976658e-01,1.044666714556024556e-01,2.499102715181559034e-02,1.700756048431492362e-01,3.016551764995864179e-02,4.832171784731574549e-02,2.426682954517205351e-02,4.458600460118083975e-02,1.587492371143390535e-01,1.463263251222290118e-01,1.128066301767671614e-01,4.499467121071250991e-02,3.782562734035258695e-02,1.188049030329687943e-02,9.487290626433914076e-02,1.157589438014280797e-01,9.087899872188892614e-02,5.000718186220513839e-02,1.244464099309604138e-01,4.180189585203544778e-02,1.136687971472823654e-02,1.007280877184962842e-01,4.855192711593218913e-02,1.003011423077787290e-01,9.002698445683551820e-02,6.430307040890140269e-02,4.700545412385447736e-02,6.403153790577391369e-02,5.453231190321245619e-02,4.981869027600405175e-02,3.830184392924008280e-02,1.737534553139372107e-02,7.136456644413785577e-02,8.472874827867805936e-02,2.949177439072193235e-02,9.298175209732851415e-03,8.307692997502660881e-02,2.748689756747571891e-02,5.466740132946840830e-02,2.091093258943841834e-02,4.088344359064395583e-02,2.922271671459738054e-02,1.338957951539661018e-01],
[7.664496837790568499e-03,1.586661871323085879e-02,2.382409230402086561e-02,1.804607062453831776e-02,1.792893251156095144e-02,2.260753873558435031e-02,3.672362242225798989e-02,1.282746810802334048e-01,1.236215668218589164e-01,2.528357102134181841e-02,1.532677526109514909e-01,1.286660821141348381e-02,3.285468516113778520e-02,2.749302153098699686e-02,4.742154900658863970e-02,1.461990327740145534e-01,1.620530612031101103e-01,1.357569361982627065e-01,3.002765506057622838e-02,4.318646072658552382e-02,2.142617986328075597e-02,9.538512496826419274e-02,7.795690889417884739e-02,6.292989298381178320e-02,5.964748723206311082e-02,1.127725410142524642e-01,5.417714981341427427e-02,3.357161134851208151e-02,1.113003395243712529e-01,5.882188031493786745e-02,8.971492768638419935e-02,7.597705093712553426e-02,7.258959380063646771e-02,4.482351910962677211e-02,6.927684644413409365e-02,5.681937993390568514e-02,5.465428711002043999e-02,2.004477279747115051e-02,2.438128510231778873e-02,6.721392842964146741e-02,8.553438990601439085e-02,1.911462804786483172e-02,3.420779714588047249e-02,6.514356270608126498e-02,1.528495233897377188e-02,3.323530181299770569e-02,1.817639508353666514e-02,6.269645053695603598e-02,1.071619538572740268e-02,1.430942262888109107e-01],
[2.550008941087052172e-02,3.669448610889066570e-02,4.880867123989044570e-02,2.692640042549566959e-02,3.656792441254478471e-02,4.548091221365325142e-03,3.930191008861182861e-02,1.245584620131276049e-01,9.784020545770502619e-02,-1.847414716675426817e-03,1.587209747740478127e-01,2.499449249670343889e-02,6.170610127645483750e-02,5.065237473338973090e-02,2.568268424875280384e-02,1.677755127138450542e-01,1.499916610584803023e-01,1.230538850039482462e-01,2.041671903162685028e-02,4.568181264522764434e-02,2.554600054006509607e-02,1.011831960712852030e-01,9.908814568791798894e-02,7.171112987635597236e-02,7.406537529070748860e-02,1.475459101352532554e-01,5.333016919504691977e-02,1.926184530659472843e-02,9.890494295465675212e-02,5.342086584312259051e-02,1.023080423741625639e-01,9.689413971661867131e-02,5.846899579745576392e-02,4.403562474453311115e-02,7.874406490462147745e-02,6.516373889377417850e-02,5.786210304308162661e-02,2.696262697494274801e-02,3.041933570858365923e-02,6.205955203643518520e-02,7.955345141063625836e-02,1.065466020742840240e-02,2.920158568843328126e-02,8.901562177553799804e-02,2.095104496892204529e-02,4.447287415717195852e-02,1.480971340359404226e-02,5.416616392937826513e-02,2.427219320154475057e-02,1.278417416141087726e-01],
[7.690486930045192619e-03,3.116625171461808591e-02,3.482690785600060790e-02,2.993469112096106693e-02,3.149834909242283515e-02,2.054217181146301280e-02,5.013271060243764538e-02,1.211482347699186501e-01,1.060704103328599240e-01,1.719155748488961424e-02,1.714086534172206411e-01,3.644230250962211121e-02,3.183873912005727447e-02,1.749338514239345832e-02,2.425799560673869495e-02,1.624891057813092066e-01,1.572689657782091466e-01,1.231517010089949599e-01,1.406313962222307395e-02,3.381182988109937559e-02,2.512012243460500296e-02,1.037480158709038480e-01,9.032490808054577380e-02,6.696368255234416700e-02,5.045030263411316024e-02,1.390881198936855245e-01,4.599976377759475848e-02,4.746364788252890030e-04,9.425163620550870702e-02,4.583320227343090636e-02,1.012591423509200395e-01,9.401741995736115354e-02,7.433067075499427689e-02,2.795719752520255813e-02,8.904155402686411702e-02,5.061795924618354287e-02,4.740165933016071004e-02,3.618489615793301500e-02,1.594172205593805472e-02,5.931611676140895184e-02,8.466464878796793603e-02,1.441280293953555759e-02,2.904672455563273720e-02,6.722251656229008554e-02,3.262104371552051324e-02,2.630400616121066071e-02,2.690908426719579652e-02,5.729555446217370063e-02,3.833011216721269460e-02,1.297592072844853950e-01],
[2.602866004842193520e-02,2.638525379270087109e-02,4.356049658071674963e-02,1.181086659649358342e-02,1.963729697099989013e-02,1.351780161426076148e-02,4.461989242955285984e-02,1.051157813380719508e-01,1.166811871299465314e-01,2.000368447475613098e-02,1.643574630118924684e-01,2.159439172358718420e-02,2.522215801135964386e-02,4.098272520672859714e-02,2.542618140691463996e-02,1.599955098184482893e-01,1.416823624834972484e-01,1.278031644805637801e-01,4.536936237679252415e-02,5.196671296596883788e-02,1.563557243071366648e-02,1.068920088450638489e-01,9.707598960150627676e-02,5.652002535372503689e-02,6.233390394822878322e-02,1.415546943662472912e-01,4.569489127179085508e-02,2.347724552173003493e-02,1.087586456698144549e-01,4.439515109207028271e-02,1.066609122990974795e-01,8.414138478218044503e-02,6.353992159830769426e-02,5.865491368345906315e-02,6.512579014345493134e-02,6.606455590331834771e-02,3.814367959893936316e-02,1.566219624074930913e-02,3.680254149845824108e-02,5.489336636075722170e-02,9.904219624337722017e-02,1.782429635681550759e-02,2.539696996237900475e-02,8.121190457695699749e-02,2.903033526276611528e-02,2.079070898861192992e-02,3.519533725741354896e-02,4.695762071490131029e-02,2.508680225230871738e-02,1.343747674773329293e-01],
[1.641900608845242235e-02,2.833740919575300499e-02,4.147410574597765881e-02,3.165302714456128680e-02,1.068609369374013199e-02,2.071910137974189783e-02,3.057551003898418759e-02,1.222399128301764615e-01,1.196766499066205525e-01,5.495968851706160190e-04,1.647453919409176681e-01,2.676982193703569129e-02,4.843317702929790525e-02,3.344094321652495844e-02,4.572652234487858958e-02,1.488433325073604541e-01,1.630999373500200411e-01,1.130526412640892875e-01,3.284525516511037357e-02,4.513797652559198292e-02,2.815179835112113782e-02,1.072994233601459735e-01,9.877144757454747737e-02,7.695905847761226604e-02,5.479747611636450605e-02,1.502663123892918340e-01,5.160326519150940788e-02,-9.214867461639255752e-04,9.429778730460096348e-02,3.634051239789847648e-02,1.077066411303536780e-01,8.535659675839457616e-02,6.453219578335708806e-02,3.703180926261710748e-02,6.816138221347428638e-02,6.342222618322916616e-02,3.492421306441723799e-02,2.985490592054755588e-02,2.339728671525213854e-02,6.722591181914813752e-02,1.032236929589123581e-01,5.890618180433695138e-03,2.416328179751232527e-02,1.001774923724668975e-01,2.091543382595851402e-02,4.927933653408111264e-02,2.296625307619650728e-03,4.849180485854527600e-02,1.679803346586147481e-02,1.288137923202719304e-01],
[8.194451512028582893e-03,3.853767155371183584e-02,3.356764003595669177e-02,4.137291260099194151e-02,1.826628031429472845e-02,1.042330497632270800e-02,4.343506759529808942e-02,1.429992991285227166e-01,1.191912137391102772e-01,2.519592466662943717e-02,1.699157143254962643e-01,2.699230622588651846e-02,6.325671662738899248e-02,3.905259385494648516e-02,2.950804630763343567e-02,1.748434626628090904e-01,1.512558680783129272e-01,1.276282931417533628e-01,3.831026046153851450e-02,4.028818204755831023e-02,2.802284442893433064e-02,1.313874517545109144e-01,8.508983546632796491e-02,5.572882260066655546e-02,5.488112880653476383e-02,1.378262060121544508e-01,6.134836626772364326e-02,6.988019697412755846e-03,1.242285749861802013e-01,3.390950232118514901e-02,1.142107412230200791e-01,1.120636079077389213e-01,7.421454729783950877e-02,5.341899869306172832e-02,5.999929436464396942e-02,6.275529633765397630e-02,4.349857234201951611e-02,4.885850593021039134e-02,2.600254397248556379e-02,5.801890775440118192e-02,8.761518544687704746e-02,1.868234162427900164e-02,2.133595130944812568e-02,7.669657609686705846e-02,2.976224730258652737e-02,5.016315644752730035e-02,2.926578764013059580e-02,4.904453407927186065e-02,1.883364766897136638e-02,1.187200413178815944e-01],
[1.979816063050904296e-02,1.928458138699712041e-02,2.322446343095594182e-02,3.621561478700114495e-02,1.849845232599055603e-02,1.356184693717030036e-02,4.340543513459679814e-02,1.253640045943443804e-01,1.053865206177960201e-01,1.589652969894934584e-02,1.556771620256998390e-01,-2.210940356410753989e-03,4.579939877444889895e-02,2.762470960594455918e-02,4.833836311069990288e-02,1.623415851080460826e-01,1.533237749451539234e-01,1.462622572269050569e-01,3.687155255662894621e-02,5.119865779106825415e-02,2.586159762340015794e-02,1.295589913394575854e-01,9.349072338522947112e-02,4.329161025510577798e-02,5.517534698585684766e-02,1.360724185962997601e-01,5.470431982080245503e-02,1.738544279489641353e-02,9.190111793076220781e-02,3.189796521887000580e-02,1.036799267005464875e-01,9.191736801576104454e-02,7.257964963447478823e-02,5.707278461566145378e-02,5.307998283551824958e-02,5.071415836316758158e-02,4.909401246434749039e-02,4.469986343798941150e-03,4.196968790632667895e-02,4.669062837626496848e-02,8.484221229073626858e-02,6.637195006881178166e-03,2.609846928984269065e-02,7.587219497014571767e-02,2.385089551831282365e-02,4.687718080781105856e-02,1.947914028640771247e-02,5.571782693685589649e-02,1.206671495766334715e-02,1.355888173243557371e-01],
[2.158550919680216063e-02,2.110563280074533032e-02,1.280721876323922571e-02,3.229782954870119571e-02,2.513191246299177617e-02,1.911182853839085705e-02,1.745340927888711438e-02,1.271270823742589595e-01,1.146303090503865835e-01,2.044517993790302690e-02,1.679791296747386864e-01,3.545254123959453666e-03,2.898957634405797956e-02,1.622758028836887925e-02,2.113942452096811281e-02,1.495463663860208403e-01,1.498613607874119957e-01,1.385592282085270632e-01,3.437578396905113798e-02,5.443582027254607186e-02,1.869144535195408838e-02,1.237887781453272801e-01,8.319866065388493992e-02,5.919165942189028723e-02,5.841180286723970938e-02,1.566795214870029540e-01,4.464338423724183852e-02,1.530058577014052970e-02,1.165583588166604762e-01,3.208116019647115624e-02,9.744712719165710135e-02,7.094321348682654238e-02,3.352220788709105864e-02,2.984109663146856939e-02,4.019512628274053007e-02,4.513536660879548940e-02,6.399218418874540182e-02,2.100417376128762073e-02,2.231606315172127500e-02,7.192837028368345709e-02,8.582938150208306582e-02,3.521528312199449784e-02,3.473149598678249711e-02,7.461288115534485399e-02,5.227214642307165260e-02,4.959252280317687223e-02,1.487742733988237434e-02,3.751987354607085046e-02,4.387753292159889823e-02,1.327535254704681200e-01],
[2.708256748694603394e-02,3.302139460163536905e-02,4.250978077882183043e-02,4.221008907418278416e-02,-2.556134857924389773e-05,9.814795274406082348e-03,2.733158525007861953e-02,1.355706606003583559e-01,1.028834282327369676e-01,1.333750538103010627e-02,1.665059888872869320e-01,2.095786259463532780e-02,4.656387681253174038e-02,4.367713190865632478e-02,5.047830947095483661e-02,1.487321153263693185e-01,1.307317595793906495e-01,1.293611518702311447e-01,3.154738525003220745e-02,6.601089822137354290e-02,5.571212029895804448e-03,1.132628704048364299e-01,8.519805946665744145e-02,7.266921875318543833e-02,6.553396688059449915e-02,1.428007214483666687e-01,5.071946726008794148e-02,2.637116391669581256e-02,1.032743708613672040e-01,4.267451641331714007e-02,1.226218176349117700e-01,6.752085474255341280e-02,6.599481419272416882e-02,5.977307747887972839e-02,6.347162079720733430e-02,6.712790614653711685e-02,6.340489044214536740e-02,4.636396916386708450e-02,1.997268174594196699e-02,6.697720206795329534e-02,1.106534949386908800e-01,9.760831207167300003e-03,2.702633326673932468e-02,7.060641356994945550e-02,2.543939145576890742e-02,4.417533563403099744e-02,1.619711749747818411e-02,3.758631243411339473e-02,5.907770192500459705e-03,1.517126513996825132e-01],
[4.012159597774792724e-02,4.793577628557733894e-02,2.467434770983091205e-02,1.287348444421006384e-02,2.302210228441013001e-02,2.179846347053407407e-02,2.608486456981751883e-02,1.362015039794544080e-01,1.192105804923161982e-01,2.364171849085492869e-02,1.508954542230828144e-01,2.302435927611888936e-02,5.632786632982073050e-02,2.351368116025558466e-02,3.274139893882940799e-02,1.458503813601614574e-01,1.617368713143347014e-01,1.235543111828978358e-01,3.517291461018604015e-02,3.148604588109145397e-02,1.648886588910176282e-02,9.515769110570949929e-02,9.444643020638282538e-02,8.735521373025875724e-02,4.490841947079950158e-02,1.226833994650188692e-01,4.953084847314460221e-02,2.475667467780437100e-02,9.750195610670955859e-02,3.970910138235570624e-02,1.103886827203461457e-01,8.815833971688420700e-02,6.240847671617224396e-02,3.220186211210938898e-02,4.939543986087904098e-02,5.789659987292056220e-02,5.924433035445194479e-02,3.616524430032852511e-02,2.781507290973522784e-02,6.414592751121556136e-02,7.933310486871310296e-02,1.789748648572290352e-02,1.095256805373984003e-02,6.239059847798303360e-02,3.794976637977346551e-03,5.777659354139309811e-02,2.760973440084143332e-02,5.690074054887288985e-02,2.660214596984695340e-02,1.422155057523278865e-01],
[2.120563716782464248e-02,2.461041772777481512e-02,3.982043917811699252e-02,2.683717026484206195e-02,2.344276666669986228e-02,1.062511648136139387e-02,2.089571806288780323e-02,1.221967746000837407e-01,9.687452121388048787e-02,1.055294012957743062e-02,1.645847838532518703e-01,4.246993649989648512e-02,4.152225350163618600e-02,2.641626701874709091e-02,4.200646690625022689e-02,1.696108383952799892e-01,1.678528465315724527e-01,1.270313282407839195e-01,4.623981314269767079e-02,4.481137022449525886e-02,4.210374725949397184e-03,1.246800389821676597e-01,1.006441394868264200e-01,6.929991267304862201e-02,5.838478993594024141e-02,1.527068272504922830e-01,5.243019444988343547e-02,7.140300301227216380e-04,1.138687923573116073e-01,5.969562265088360076e-02,1.229430943718398395e-01,9.603927525411257160e-02,6.733042775327978402e-02,3.752046538410203774e-02,6.520406466516698307e-02,5.642499681721239307e-02,3.923789274139006383e-02,2.333413247447749250e-02,2.658255568282383990e-02,6.375128244510701314e-02,8.956985932949355911e-02,3.651571707260061611e-02,1.833965685175089483e-02,9.573679227302994699e-02,4.044782822376106340e-02,4.160022432522687136e-02,9.138313193407590629e-03,5.867642275143486463e-02,8.005198580875687656e-03,1.591706279940046653e-01]]).reshape((15,50))

    # W2 is the weight matrix occuring between the hidden layer and output layer. Dimension is 1x15
    # W2 is learnt using Temporal Difference learning after playing the agent against itself 100,000 times     
    W2 = np.asarray([1.419154625509130074e+00,1.447851834643198554e+00,1.434543008678328402e+00,1.443704508829750655e+00,1.411903033764211557e+00,1.442231736698699640e+00,1.420888952349470324e+00,1.434834976545719787e+00,1.446215085567697400e+00,1.468379577471502939e+00,1.425963757923194075e+00,1.426711376509014917e+00,1.443056953404695752e+00,1.426887584341695270e+00,1.469933740954004753e+00]).reshape((1,15))

    X  = np.reshape(game._board_state[0:-2],(50,1)) # Get the status of the 49 cells plus the player to play
    z1 = np.dot(W1, X)                              # Dot product of inputs and weights of first layer
    a1 = expit(z1)                                  # Sigmoid transformation
    z2 = np.dot(W2, a1)                             # Dot product of hidden layer output with weights of second layer 
    a2 = expit(z2)                                  # Sigmoid transformation
    score = a2                                      # Score will be between 0 and 1 indicating the probability of winning from the state
    
    return score[0][0]

    # TODO: finish this function!
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    from math import sqrt                   # additional python modules
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)       # Get the opponent
    w, h = game.width / 2., game.height / 2.   # Get the coordinates of board's center

    # weights : An array of weights. Each weight is associated with one of the features.
    # weights are learnt by using Genetic algorithm. 
    # Population of agents are created with each agent associated with a set of random weights
    # Then by way of mutation and breeding for various generations, we arrive at the optimal set of weights
    weights = [-1.2105989375180854, 1.2398074905752983, -1.8746867409123202, 2.034297986274559, 0.960368159892905]

    p0 = len(game.get_legal_moves(opponent))   # Feature : No of moves available to the opponent
    p1 = len(game.get_legal_moves(player))     # Feature : No of moves available to the player

    y, x = game.get_player_location(player)    # Get player's location
    p2 =  sqrt((h - y)**2 + (w - x)**2)        # Feature : Distance of the player from board's center

    y, x = game.get_player_location(opponent)  # Get opponent's position
    p3 =  sqrt((h - y)**2 + (w - x)**2)        # Feature : Distance of the opponent from board's center
        
    p4 = len(game._board_state[0:-3])-sum(game._board_state[0:-3])   # Feature : Number of cells already used
    
    score = weights[0]*p0 + weights[1]*p1 + weights[2]*p2 + weights[3]*p3 + weights[4]*p4

    return score
    
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Uses the property of intersection. 
    # Drives the player to occupy cells such that opponent's options are reduced.
    own_moves = set(game.get_legal_moves(player))                      # Set of moves available to the player
    opp_moves = set(game.get_legal_moves(game.get_opponent(player)))   # Set of moves available to the opponent
    cmn_moves = own_moves.intersection(opp_moves)                      # Set of moves common to both the players
    
    return float(len(own_moves) - len(opp_moves) + len(cmn_moves))     # Drive the player to the move that will reduce opponent's options
    
    # TODO: finish this function!
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = None

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        #assert(best_move is not None)
        return best_move
    
    def minValue(self, game, depth):
        """Opponent's turn. 
        Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int 
            A strictly positive integer (i.e., 1, 2, 3,...) indicating the number of
            depth of the current state in the game tree

        Returns
        -------
        (int, (int, int))
            Value and Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        v = float("inf")
        m = (-1, -1)
    
        if(depth == 0):                                  # Have reached the maximum depth allowed
            v = self.score(game, game._inactive_player)  # Get the score at the current state
        else:
            succMoves = game.get_legal_moves()           # Get all the successors of the current state/node
            if(len(succMoves)==0):                       # If no successors, it implies we have reached leaf node
                v = float('inf')                         # If opponent has no moves, the player will win. Return max value.
            else:
                for move in succMoves:                                 # Try every successor
                    succGame = game.forecast_move(move)                # Make the move
                    valueOfMove,tmp = self.maxValue(succGame, depth-1) # Get the worth of the move
                    if(valueOfMove <= v):                              # If the move is better than what we have seen so far,
                        v = valueOfMove                                # Save the value
                        m = move                                       # Save the move
                    # end if
                # end for
            # end if
        # end if
        return (v,m)                                     # Return the value and the best move that the opponent will make
        
    def maxValue(self, game, depth):
        """Player's turn. 
        Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int 
            A strictly positive integer (i.e., 1, 2, 3,...) indicating the number of
            depth of the current state in the game tree

        Returns
        -------
        (int, (int, int))
            Value and Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        v = float("-inf")
        m = (-1, -1)

        if(depth == 0):                                # Have reached the maximum depth allowed
            v =  self.score(game, game._active_player) # Get the score at the current state
        else:
            succMoves = game.get_legal_moves()         # Get all the successors of the current state/node
            if(len(succMoves)==0):                     # If no successors, it implies we have reached leaf node
                v = float('-inf')                      # If player has no moves, then opponent will win. Return least value.
            else:
                for move in succMoves:                                 # Try every successor
                    succGame = game.forecast_move(move)                # Make the move
                    valueOfMove,tmp = self.minValue(succGame, depth-1) # Get the worth of the move
                    if(valueOfMove >= v):                              # If the move is better than what we have seen so far,
                        v = valueOfMove                                # Save the value
                        m = move                                       # Save the move
                    # end if
                # end for
            # end if
        # end if      
        return (v,m)                                   # Return the value and the best move that the player will make
    
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        value, move = self.maxValue(game,depth)

        return move
        # TODO: finish this function!
        raise NotImplementedError


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        depth = self.search_depth    # Depth to search for the first time
        best_move = (-1, -1)
        try: # Iterative deepening search
            while(self.time_left() > self.TIMER_THRESHOLD):                          # Check if we have adequate time left
                best_move = self.alphabeta(game, depth, float("-inf"), float("inf")) # search till depth
                depth = depth+1                                                      # Increase the depth for the next iteration
        except SearchTimeout:            
            pass
        return best_move
        # TODO: finish this function!
        raise NotImplementedError

    def minValue(self, game, depth, alpha, beta):
        """ Opponent's turn.
        Search for the best move from the available legal moves and return a
        result before the time limit expires. 
        Use alpha-beta pruning to cut down the branches to search

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the current depth of the game tree

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, (int, int))
            Value and the board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        v = float("inf")
        m = (-1, -1)
    
        if(depth == 0):                                 # Have reached the maximum depth allowed
            v = self.score(game, game._inactive_player) # Get the score at the current state
        else:
            succMoves = game.get_legal_moves()          # Get all the successors of the current state/node
            if(len(succMoves) == 0):                    # If no successors, it implies we have reached leaf node
                v = float('inf')                        # If opponent has no moves, then player will win. Return max value.
            else:    
                for move in succMoves:                                              # Try every successor
                    succGame = game.forecast_move(move)                             # Make the move
                    valueOfMove,tmp = self.maxValue(succGame, depth-1, alpha, beta) # Get the worth of the move
                    if(valueOfMove <= v):                                           # If the move is better than what we have seen so far,
                        v = valueOfMove                                             # Save the value
                        m = move                                                    # Save the move
                    if(v <= alpha):                                                 # Check if we can stop our search at this branch itself
                        break
                    beta = min(v, beta)                                             # Update the beta value
                # end for
            # end if
        # end if       
        return (v, m)                                   # Return the value and best move that the opponent will make
        
    def maxValue(self, game, depth, alpha, beta):
        """ Players's turn.
        Search for the best move from the available legal moves and return a
        result before the time limit expires. 
        Use alpha-beta pruning to cut down the branches to search

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the current depth of the game tree

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, (int, int))
            Value and the board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        v = float("-inf")
        m = (-1, -1)
    
        if(depth == 0):                                # Have reached the maximum depth allowed
            v =  self.score(game, game._active_player) # Get the score at the current state
        else:
            succMoves = game.get_legal_moves()         # Get all the successors of the current state/node
            if(len(succMoves) == 0):                   # If no successors, it implies we have reached leaf node               
                v = float('-inf')                      # If player has no moves, then opponent will win. Return least value.
            else:
                for move in succMoves:                                              # Try every successor
                    succGame = game.forecast_move(move)                             # Make the move
                    valueOfMove,tmp = self.minValue(succGame, depth-1, alpha, beta) # Get the worth of the move
                    if(valueOfMove >= v):                                           # If the move is better than what we have seen so far,
                        v = valueOfMove                                             # Save the value
                        m = move                                                    # Save the move
                    if(v >= beta):                                                  # Check if we can stop our search at this branch itself
                        break
                    alpha = max(v, alpha)                                           # Update the alpha value                    
                # end for
            # end if
        # end if        
        return (v, m)                                 # Return the value and best move that the player will make

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        value, move = self.maxValue(game, depth, alpha, beta)            
        return move
        # TODO: finish this function!
        raise NotImplementedError
